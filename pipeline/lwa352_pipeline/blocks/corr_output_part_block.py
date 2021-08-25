import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap
from bifrost.ndarray import copy_array
from bifrost.device import stream_synchronize, set_device as BFSetGPU
from bifrost.udp_socket import UDPSocket
from bifrost.packet_writer import HeaderInfo, UDPTransmit
from bifrost.address import Address

import time
import ujson as json
import socket
import struct
import numpy as np

from .block_base import Block

COR_NPOL = 2

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
        | ``nchan_sum`` | int    |       | The number of frequency channels summed by     |
        |               |        |       | upstream processing                            |
        +---------------+--------+-------+------------------------------------------------+
        | ``chan0``     | int    |       | The index of the first frequency channel in    |
        |               |        |       | the input visibility matrices                  |
        +---------------+--------+-------+------------------------------------------------+
        | ``npol``      | int    |       | The number of polarizations per stand in the   |
        |               |        |       | input visibility matrices                      |
        +---------------+--------+-------+------------------------------------------------+
        | ``bw_hz``     | double | Hz    | Bandwidth of the input visibility matrices.    |
        +---------------+--------+-------+------------------------------------------------+
        | ``sfreq``     | double | Hz    | Center frequency of the first channel in the   |
        |               |        |       | input visibility matrices. Only required if    |
        |               |        |       | ``use_cor_fmt=False``.                         |
        +---------------+--------+-------+------------------------------------------------+
        | ``fs_hz``     | int    | Hz    | ADC sample rate. Only required if              |
        |               |        |       | ``use_cor_fmt=True``                           |
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
        the custom format (i.e., if ``use_cor_fmt=False``). If using the COR format, this
        parameter has no effect.
    :type nvis_per_packet: int

    :param npipeline: The number of pipelines in the system, as written to output COR packets.
        This may or may not be the same as the number of actual pipelines present. The one-index of
        this block's pipeline is computed from sequence header values as
        ``((chan0 // nchan) % npipeline) + 1``.
    :type npipeline: int

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

    If ``use_cor_fmt=True``, this block outputs packets conforming to the LWA-SV "COR"
    spec (though with integer rather than floating point data). This format comprises
    a stream of UDP packets, each with a 32 byte header defined as follows:

    .. code:: C
    
          struct cor {
            uint32_t  sync_word;
            uint8_t   id;
            uint24_t  frame_number;
            uint32_t  secs_count;
            int16_t   freq_count;
            int16_t   cor_gain;
            int64_t   time_tag;
            int32_t   cor_navg;
            int16_t   stand_i;
            int16_t   stand_j;
            int32_t   data[nchans, npols, npols, 2];
          };

    Packet fields are as follows:

    .. table::
        :widths: 25 10 10 55

        +---------------+------------+----------------+---------------------------------------------+
        | Field         | Format     | Units          | Description                                 |
        +===============+============+================+=============================================+
        | sync\_word    | uint32     |                | Mark 5C magic number, ``0xDEC0DE5C``        |
        +---------------+------------+----------------+---------------------------------------------+
        | id            | uint8      |                | Mark 5C ID, used to identify COR packet,    |
        |               |            |                | ``0x02``                                    |
        +---------------+------------+----------------+---------------------------------------------+
        | frame_number  | uint24     |                | Mark 5C frame number.  Used to store info.  |
        |               |            |                | about how to order the output packets.  The |
        |               |            |                | first 8 bits contain the channel decimation |
        |               |            |                | fraction relative to the F-Engine output.   |
        |               |            |                | The next 8 bits contain the total number of |
        |               |            |                | subbands being transmitted.  The final 8    |
        |               |            |                | contain which subband is contained in the   |
        |               |            |                | packet.                                     |
        +---------------+------------+----------------+---------------------------------------------+
        | secs_count    | uint32     |                | Mark 5C seconds since 1970-01-01 00:00:00   |
        |               |            |                | UTC. Unused.                                |
        +---------------+------------+----------------+---------------------------------------------+
        | freq_count    | int16      |                | zero-indexed frequency channel ID of the    |
        |               |            |                | first channel in the packet.                |
        +---------------+------------+----------------+---------------------------------------------+
        | cor_gain      | int16      |                | Right bitshift used for gain compensation.  |
        |               |            |                | Unused.                                     |
        +---------------+------------+----------------+---------------------------------------------+
        | time_tag      | int64      | ADC sample     | Central sampling time since 1970-01-01      |
        |               |            | period         | 00:00:00 UTC.                               |
        +---------------+------------+----------------+---------------------------------------------+
        | cor_navg      | int16      | TODO: subslots | Integration time.                           |
        |               |            | doesn't work   |                                             |
        +---------------+------------+----------------+---------------------------------------------+
        | stand_i       | int16      |                | 1-indexed stand number of the unconjugated  |
        |               |            |                | stand.                                      |
        +---------------+------------+----------------+---------------------------------------------+
        | stand_j       | int16      |                | 1-indexed stand number of the conjugated    |
        |               |            |                | stand.                                      |
        +---------------+------------+----------------+---------------------------------------------+
        | data          | int32\*    |                | The data payload. Data for the visibility   |
        |               |            |                | of antennas at stand_i and stand_j, with    |
        |               |            |                | stand_j conjugated. Data are a              |
        |               |            |                | multidimensional array of 32-bit integers,  |
        |               |            |                | with dimensions ``[nchans, npols, npols,    |
        |               |            |                | 2]``. The first axis is frequency channel.  |
        |               |            |                | The second axis is the polarization of the  |
        |               |            |                | antenna at stand_i. The second axis is the  |
        |               |            |                | polarization of the antenna at stand_j.|    |
        |               |            |                | The fourth axis is complexity, with index 0 |
        |               |            |                | the real part of the visibility, and index  |
        |               |            |                | 1 the imaginary part.                       |
        +---------------+------------+----------------+---------------------------------------------+

    If ``use_cor_fmt=False``:

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
            int32_t   data[nvis, nchans, 2];
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
                 guarantee=True, core=-1, etcd_client=None, dest_port=10001, nvis_per_packet=16, 
                 nchan_sum=1, pipeline_idx=1, npipeline=1):
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        super(CorrOutputPart, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)

        self.nvis_per_packet = nvis_per_packet
        self.nchan_sum = nchan_sum
        self.pipeline_idx = pipeline_idx
        self.npipeline = npipeline

        # Do this now since it doesn't change after the block is initialized
        wrapped_idx = ((self.pipeline_idx - 1) % self.npipeline) + 1
        self.tuning = (self.nchan_sum << 16) | (self.npipeline << 8) | wrapped_idx
        self.tuning &= 0x00FFFFFF
        
        self.use_cor_fmt = use_cor_fmt
        if self.use_cor_fmt:
            self.sock = None
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.01)

        self.define_command_key('dest_ip', type=str, initial_val='0.0.0.0')
        self.define_command_key('dest_port', type=int, initial_val=dest_port)
        self.update_command_vals()

    def send_packets_py(self, dout, baselines, sync_time, this_gulp_time, bw_hz, sfreq,
                              upstream_acc_len, nchan, chan0):
        cpu_affinity.set_core(self.core)
        baselines_flat = baselines.flatten()
        for vn in range(len(baselines)//self.nvis_per_packet):
            header = struct.pack(">QQ2d4I",
                                 sync_time,
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
            self.sock.sendto(header + dout[vn*self.nvis_per_packet : (vn+1)*self.nvis_per_packet].tobytes(),
                                             (self.command_vals['dest_ip'], self.command_vals['dest_port']))

    def send_packets_bf(self, data, baselines, udt, time_tag, desc, chan0, nchan, gain, navg):
        cpu_affinity.set_core(self.core)
        start_time = time.time()
        desc.set_chan0(chan0)
        desc.set_gain(gain)
        desc.set_decimation(navg)
        nvis_per_pkt = COR_NPOL**2 # 4 baselines per packet because assume dual pol
        nvis = len(baselines)
        desc.set_nsrc(nvis // nvis_per_pkt)
        nstand_virt = int((-1 + np.sqrt(1 + 2*nvis))/2) # effective number of stands
        desc.set_tuning(self.tuning)
        pkt_payload_bits = nchan * nvis_per_pkt * 8 * 8
        start_time = time.time()
        dview = data.view('cf32').reshape([nchan, nvis // nvis_per_pkt, COR_NPOL, COR_NPOL])
        first_baseline = 0
        baselines_sent = 0
        for i in range(nstand_virt):
            # `data` should be sent in order stand1 x stand0 x chan x pol1 x pol0 x complexity
            # Input view is nchan x baselines x pol1 x pol0. Assume they are ordered
            # So that all the stand0 baselines come first, then stand1, etc.
            # Read a single stand
            n_bl_this_stand = nstand_virt - i
            last_baseline = first_baseline + n_bl_this_stand
            sdata = dview[:, first_baseline:last_baseline, :, :].copy(space='system')
            # reshape and send
            sdata = sdata.transpose([1,0,2,3]).reshape(1, -1, nchan*nvis_per_pkt).view('cf32')
            udt.send(desc, time_tag, 0, first_baseline, 1, sdata)
            first_baseline += sdata.shape[1]
            baselines_sent += n_bl_this_stand
        del sdata
        assert first_baseline == baselines_sent
        assert baselines_sent == nvis // 4
        stop_time = time.time()
        elapsed = stop_time - start_time
        gbps = 8 * 8 * nvis * nchan / elapsed / 1e9
        self.update_stats({'output_gbps': gbps})

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        desc = HeaderInfo()
        prev_time = time.time()
        for iseq in self.iring.read(guarantee=self.guarantee):
            self.update_pending = True # Reprocess commands on each new sequence
            udt = None
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            upstream_acc_len = ihdr['acc_len']
            upstream_start_time = this_gulp_time
            baselines = np.array(ihdr['baselines'], dtype='>i')
            baselines_flat = baselines.flatten()
            nchan = ihdr['nchan']
            nchan_sum = ihdr['nchan_sum']
            chan0 = ihdr['chan0']
            bw_hz = ihdr['bw_hz']
            nvis  = ihdr['nvis']
            if not self.use_cor_fmt:
                sfreq = ihdr['sfreq']
            if self.use_cor_fmt:
                samples_per_spectra = int(nchan_sum * nchan * ihdr['fs_hz'] / bw_hz)
            igulp_size = nvis * nchan * 8
            dout = np.zeros(shape=[nvis, nchan, 2], dtype='>i')
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # skip last gulp
                # Update destinations if necessary
                if self.update_pending:
                    self.update_command_vals()
                    self.log.info("CORR PART OUTPUT >> Updating destination to %s:%s" %
                                  (self.command_vals['dest_ip'], self.command_vals['dest_port']))
                    if self.use_cor_fmt:
                        if self.sock is None:
                            self.sock = UDPSocket()
                        else:
                            self.sock.close()
                        self.sock.connect(Address(self.command_vals['dest_ip'], self.command_vals['dest_port']))
                        if not isinstance(udt, UDPTransmit):
                            udt = UDPTransmit('cor_%i' % nchan, sock=self.sock, core=self.core)
                self.stats.update({'curr_sample': this_gulp_time})
                self.update_stats()
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                if self.command_vals['dest_ip'] != "0.0.0.0":
                    if self.use_cor_fmt:
                        time_tag = this_gulp_time * samples_per_spectra
                        # Read chan x baseline x complexity input data.
                        idata = ispan.data_view('i32').reshape([nchan, nvis, 2])
                        self.send_packets_bf(idata, baselines, udt, time_tag, desc, chan0, nchan, 0,
                                upstream_acc_len * samples_per_spectra)
                    else:
                        # Read chan x baseline x complexity input data.
                        # Transpose to baseline x chan x complexity
                        idata = ispan.data_view('i32').reshape([nchan, nvis, 2]).transpose([1,0,2])
                        # Do an actual copy so that we have binary data formatted for sending
                        dout[...] = idata;
                        self.send_packets_py(dout, baselines, ihdr['sync_time'], this_gulp_time,
                                             bw_hz, sfreq, upstream_acc_len, nchan, chan0)
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.update_stats()
                # And, update overall time counter
                this_gulp_time += upstream_acc_len
