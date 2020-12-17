import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap
from bifrost.ndarray import copy_array
from bifrost.device import stream_synchronize, set_device as BFSetGPU

import time
import simplejson as json
import socket
import struct
import numpy as np

from .block_base import Block


class CorrOutputPart(Block):
    """
    **Functionality**

    Output a block of visibilities as a UDP data stream

    **New Sequence Condition**

    This block is a bifrost sink, and generates no downstream sequences.

    **Input Header Requirements**

    This block requires that the following header fields
    be provided by the upstream data source:

    .. table::
        :widths: 25 10 10 55

        +---------------+--------+-------+------------------------------------------------+
        | Field         | Format | Units | Description                                    |
        +===============+========+=======+================================================+
        | ``seq0``      | int    |       | Spectra number for the first sample in the     |
        |               |        |       | input sequence                                 |
        +---------------+--------+-------+------------------------------------------------+
        | ``acc_len``   | int    |       | Number of spectra integrated into each output  |
        |               |        |       | sample by upstream processing                  |
        +---------------+--------+-------+------------------------------------------------+
        | ``nchan``     | int    |       | The number of frequency channels in the input  |
        |               |        |       | visibility matrices                            |
        +---------------+--------+-------+------------------------------------------------+
        | ``chan0``     | int    |       | The index of the first frequency channel in    |
        |               |        |       | the input visibility matrices                  |
        +---------------+--------+-------+------------------------------------------------+
        | ``npol``      | int    |       | The number of polarizations per stand in the   |
        |               |        |       | input visibility matrices                      |
        +---------------+--------+-------+------------------------------------------------+
        | ``bw_hz``     | double | Hz    | Bandwidth of the input visibility matrices.    |
        |               |        |       | Only required if ``use_cor_fmt=False``         |
        +---------------+--------+-------+------------------------------------------------+
        | ``sfreq``     | double | Hz    | Center frequency of the first channel in the   |
        |               |        |       | input visibility matrices. Only required if    |
        |               |        |       | ``use_cor_fmt=False``.                         |
        +---------------+--------+-------+------------------------------------------------+
        | ``nvis``      | int    | -     | Number of visibilities in the output data      |
        |               |        |       | stream                                         |
        +---------------+--------+-------+------------------------------------------------+
        | ``baselines`` | list   | -     | A list of output stand/pols, with dimensions   |
        |               | of     |       | ``[nvis, 2, 2]``. E.g. if entry :math:`[V]` of |
        |               | ints   |       | this list has value ``[[N_0, P_0], [N_1,       |
        |               |        |       | P_1]]`` then the ``V``-th entry in the output  |
        |               |        |       | data array is the correlation of stand         |
        |               |        |       | ``N_0``, polarization ``P_0`` with stand       |
        |               |        |       | ``N_1``, polarization ``P_1``                  |
        +---------------+--------+-------+------------------------------------------------+

    **Output Headers**

    This is a bifrost sink block, and provides no data to an output ring.

    **Data Buffers**

    *Input Data Buffer*: A CPU-side bifrost ring buffer with 32+32 bit complex integer data
    in order ``time x channel x visibility x complexity``
    This input buffer is read in gulps of ``nchan x nvis`` words, each 8 bytes in size.

    *Output Data Buffer*: This block has no output data buffer.

    **Instantiation**

    :param log: Logging object to which runtime messages should be
        emitted.
    :type log: logging.Logger

    :param iring: bifrost input data ring. This should be on the GPU.
    :type iring: bifrost.ring.Ring

    :param guarantee: If True, read data from the input ring in blocking "guaranteed"
        mode, applying backpressure if necessary to ensure no data are missed by this block.
    :type guarantee: Bool

    :param core: CPU core to which this block should be bound. A value of -1 indicates no binding.
    :type core: int

    :parameter etcd_client: Etcd client object used to facilitate control of this block.
        If ``None``, do not use runtime control.
    :type etcd_client: etcd3.client.Etcd3Client

    :param dest_port: Default destination port for UDP data. Can be overriden with the runtime
        control interface.
    :type dest_port: int

    :param use_cor_fmt: If True, use the LWA ``COR`` packet output format. Otherwise use a custom
        format. See *Output Format*, below. Currently, only ``use_cor_fmt=False`` is supported.
    :type use_cor_fmt: Bool

    :param nvis_per_packet: Number of visibilities to pack into a single UDP packet, if using
        the custom format (i.e., if ``use_core_fmt=False``).
    :type nvis_per_packet: int

    **Runtime Control and Monitoring**

    .. table::
        :widths: 25 10 10 55

        +------------------+--------+---------+------------------------------+
        | Field            | Format | Units   | Description                  |
        +==================+========+=========+==============================+
        | ``dest_ip``      | string |         | Destination IP for           |
        |                  |        |         | transmitted packets, in      |
        |                  |        |         | dotted-quad format. Eg.      |
        |                  |        |         | ``"10.0.0.1"``. Use          |
        |                  |        |         | ``"0.0.0.0"`` to skip        |
        |                  |        |         | sending packets              |
        +------------------+--------+---------+------------------------------+
        | ``dest_port``    | int    |         | UDP port to which packets    |
        |                  |        |         | should be transmitted.       |
        +------------------+--------+---------+------------------------------+

    **Output Data Format**

    The ``COR`` packet format is not yet supported. The below description
    details the packet format when ``use_cor_fmt=False``.

    Each packet from the correlator contains data from multiple channels for multiple
    single-polarization baselines. There are two possible output formats depending on the
    value of ``use_cor_fmt`` with which this block is instantiated.

    If ``use_cor_fmt=False``, this block outputs a stream of UDP packets, with each comprising
    a 56 byte header followed by a payload of signed 32-bit integers. The packet definition is
    as follows:

    .. code:: C
    
          struct corr_output_partial_packet {
            uint64_t  sync_time;
            uint64_t  spectra_id;
            double    bw_hz;
            double    sfreq_hz;
            uint32_t  acc_len;
            uint32_t  nvis;
            uint32_t  nchans;
            uint32_t  chan0;
            uint32_t  baselines[nvis, 2, 2];
            int32_t   data[npols, npols, nchans, 2];
          };

    Packet fields are as follows:

    .. table::
        :widths: 25 10 10 55

        +---------------+---------------------+----------------+---------------------------------------------+
        | Field         | Format              | Units          | Description                                 |
        +===============+=====================+================+=============================================+
        | sync\_time    | uint64              | UNIX seconds   | The sync time to which spectra IDs are      |
        |               |                     |                | referenced.                                 |
        +---------------+---------------------+----------------+---------------------------------------------+
        | spectra\_id   | int                 | -              | The spectrum number for the first spectra   |
        |               |                     |                | which contributed to this packet’s          |
        |               |                     |                | integration.                                |
        +---------------+---------------------+----------------+---------------------------------------------+
        | bw\_hz        | double (binary64)   | Hz             | The total bandwidth of data in this packet  |
        +---------------+---------------------+----------------+---------------------------------------------+
        | sfreq\_hz     | double (binary64)   | Hz             | The center frequency of the first channel   |
        |               |                     |                | of data in this packet                      |
        +---------------+---------------------+----------------+---------------------------------------------+
        | acc\_len      | uint32              | -              | The number of spectra integrated in this    |
        |               |                     |                | packet                                      |
        +---------------+---------------------+----------------+---------------------------------------------+
        | nvis          | uint32              | -              | The number of single polarization           |
        |               |                     |                | visibilities present in this packet.        |
        +---------------+---------------------+----------------+---------------------------------------------+
        | nchans        | uint32              | -              | The number of frequency channels in this    |
        |               |                     |                | packet. For LWA-352 this is 184             |
        +---------------+---------------------+----------------+---------------------------------------------+
        | chan0         | uint32              | -              | The index of the first frequency channel in |
        |               |                     |                | this packet                                 |
        +---------------+---------------------+----------------+---------------------------------------------+
        | baselines     | uint32\*            | -              | An array containing the stand and           |
        |               |                     |                | polarization indices of the multiple        |
        |               |                     |                | visibilities present in this packet. This   |
        |               |                     |                | entry has dimensions [``nvis``, 2, 2]. The  |
        |               |                     |                | first index runs over the number of         |
        |               |                     |                | visibilities within this packet. The second |
        |               |                     |                | index is 0 for the first (unconjugated)     |
        |               |                     |                | visibility input and 1 for the second       |
        |               |                     |                | (conjugated) antenna input. The third index |
        |               |                     |                | is zero for stand number, and 1 for         |
        |               |                     |                | polarization number.                        |
        +---------------+---------------------+----------------+---------------------------------------------+
        | data          | int32\*             | -              | The data payload. Data for the visibility   |
        |               |                     |                | of antennas at stand0 and stand1, with      |
        |               |                     |                | stand1 conjugated. Data are a               |
        |               |                     |                | multidimensional array of 32-bit integers,  |
        |               |                     |                | with dimensions [``nvis``, ``nchans``, 2].  |
        |               |                     |                | The first axis runs over the multiple       |
        |               |                     |                | visibilities in this packet. Each index can |
        |               |                     |                | be associated with a physical antenna using |
        |               |                     |                | the ``baselines`` field. The second axis is |
        |               |                     |                | frequency channel. The third axis is        |
        |               |                     |                | complexity, with index 0 the real part of   |
        |               |                     |                | the visibility, and index 1 the imaginary   |
        |               |                     |                | part.                                       |
        +---------------+---------------------+----------------+---------------------------------------------+
    """

    def __init__(self, log, iring, use_cor_fmt=False,
                 guarantee=True, core=-1, etcd_client=None, dest_port=10001, nvis_per_packet=16):
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        super(CorrOutputPart, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)

        self.nvis_per_packet = nvis_per_packet

        self.use_cor_fmt = use_cor_fmt
        if self.use_cor_fmt:
            self.sock = None
            raise NotImplementedError
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.01)

        self.dest_ip = "0.0.0.0"
        self.new_dest_ip = "0.0.0.0"
        self.dest_port = dest_port
        self.new_dest_port = dest_port
        self.update_pending = True


    def _etcd_callback(self, watchresponse):
        """
        A callback to run whenever this block's command key is updated.
        Decodes integration start time and accumulation length and
        preps to update the pipeline at the end of the next integration.
        """
        v = json.loads(watchresponse.events[0].value)
        if 'dest_ip' in v:
            self.new_dest_ip = v['dest_ip']
        if 'dest_port' in v:
            self.new_dest_port = v['dest_port']
        self.update_pending = True
        self.stats.update({'new_dest_ip': self.new_dest_ip,
                           'new_dest_port': self.new_dest_port,
                           'update_pending': self.update_pending,
                           'last_cmd_time': time.time()})
        self.update_stats()

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        prev_time = time.time()
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            upstream_acc_len = ihdr['acc_len']
            upstream_start_time = this_gulp_time
            baselines = np.array(ihdr['baselines'], dtype='>i')
            baselines_flat = baselines.flatten()
            nchan = ihdr['nchan']
            chan0 = ihdr['chan0']
            bw_hz = ihdr['bw_hz']
            sfreq = ihdr['sfreq']
            nvis  = ihdr['nvis']
            igulp_size = nvis * nchan * 8
            dout = np.zeros(shape=[nvis, nchan, 2], dtype='>i')
            udt = None
            for ispan in iseq.read(igulp_size):
                # Update destinations if necessary
                if self.update_pending:
                    self.dest_ip = self.new_dest_ip
                    self.dest_port = self.new_dest_port
                    self.update_pending = False
                    self.log.info("CORR PART OUTPUT >> Updating destination to %s:%s" % (self.dest_ip, self.dest_port))
                    self.stats.update({'dest_ip': self.dest_ip,
                                       'dest_port': self.dest_port,
                                       'update_pending': self.update_pending,
                                       'last_update_time': time.time()})
                    if self.use_cor_fmt:
                        if self.sock: del self.sock
                        if udt: del udt
                        self.sock = UDPSocket()
                        self.sock.connect(Address(self.dest_ip, self.dest_port))
                        
                        udt = UDPTransmit('cor_%i' % self.nchan, sock=self.sock, core=self.core)
                        desc = HeaderInfo()
                self.stats.update({'curr_sample': this_gulp_time})
                self.update_stats()
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                if self.dest_ip != "0.0.0.0":
                    idata = ispan.data_view('i32').reshape([nchan, nvis, 2]).transpose([1,0,2]) # baseline x chan x complexity
                    dout[...] = idata;
                    if self.use_cor_fmt:
                        pass
                    else:
                        for vn in range(len(baselines)//self.nvis_per_packet):
                            header = struct.pack(">QQ2d4I",
                                                 ihdr['sync_time'],
                                                 this_gulp_time,
                                                 bw_hz,
                                                 sfreq,
                                                 upstream_acc_len,
                                                 self.nvis_per_packet,
                                                 nchan,
                                                 chan0,
                                                 ) + baselines_flat[vn*4*self.nvis_per_packet : (vn+1)*4*self.nvis_per_packet].tobytes()
                            #print(baselines_flat[vn*4*self.nvis_per_packet : (vn+1)*4*self.nvis_per_packet])
                            #print(dout[vn*self.nvis_per_packet : (vn+1)*self.nvis_per_packet])
                            self.sock.sendto(header + dout[vn*self.nvis_per_packet : (vn+1)*self.nvis_per_packet].tobytes(), (self.dest_ip, self.dest_port))
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.stats.update({'last_end_sample': this_gulp_time})
                self.update_stats()
                # And, update overall time counter
                this_gulp_time += upstream_acc_len
