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
from bifrost.packet_writer import HeaderInfo, DiskWriter, UDPTransmit
from bifrost.address import Address

import os
import time
import ujson as json
import socket
import struct
import numpy as np

from .block_base import Block

class CorrOutputFull(Block):
    """
    **Functionality**

    Output an xGPU-spec visibility buffer as a stream of UDP packets.

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

    Optional header fields, which describe the input xGPU buffer contents. If not
    supplied as headers, these should be provided as keyword arguments when this
    block is instantiated.

    .. table::
        :widths: 25 10 10 55

        +------------------+--------+-------+------------------------------------------------+
        | Field            | Format | Units | Description                                    |
        +==================+========+=======+================================================+
        | ``ant_to_        | list   |       | A 4D list of integers, with dimensions         |
        | bl_id``          | of int |       | ``[nstand, nstand, npol, npol]`` which maps    |
        |                  |        |       | the correlation of ``stand0, pol0`` with       |
        |                  |        |       | ``stand1, pol1`` to visibility index           |
        |                  |        |       | ``[stand0, stand1, pol0, pol1]``               |
        +------------------+--------+-------+------------------------------------------------+
        | ``bl_is_conj``   | list   |       | A 4D list of boolean values, with dimensions   |
        |                  | of     |       | ``[nstand, nstand, npol, npol]`` which         |
        |                  | bool   |       | indicates if the correlation of ``stand0,      |
        |                  |        |       | pol0`` with ``stand1, pol1`` has the first     |
        |                  |        |       | (``stand0, pol0``) or second (``stand1,        |
        |                  |        |       | pol1``) input conjugated. If                   |
        |                  |        |       | ``bl_id_conj[stand0, stand1, pol0, pol1]`` has |
        |                  |        |       | the value ``True``, then ``stand0,pol0`` is    |
        |                  |        |       | the conjugated input.                          |
        +------------------+--------+-------+------------------------------------------------+

    **Output Headers**

    This is a bifrost sink block, and provides no data to an output ring.

    **Data Buffers**

    *Input Data Buffer*: A CPU-side bifrost ring buffer of 32+32 bit complex integer data.
    This input buffer is read in gulps of ``nchan * (nstand//2+1)*(nstand//4)*npol*npol*4*2``
    32-bit words, which is the size of an xGPU visibility matrix.

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

    :param nchan: Number of frequency channels per time sample.
    :type nchan: int

    :param nstand: Number of stands per time sample.
    :type nstand: int

    :param npol: Number of polarizations per stand.
    :type npol: int

    :parameter etcd_client: Etcd client object used to facilitate control of this block.
        If ``None``, do not use runtime control.
    :type etcd_client: etcd3.client.Etcd3Client

    :param dest_port: Default destination port for UDP data. Can be overriden with the runtime
        control interface.
    :type dest_port: int

    :param use_cor_fmt: If True, use the LWA ``COR`` packet output format. Otherwise use a custom
        format. See *Output Format*, below.
    :type use_cor_fmt: Bool

    :param antpol_to_bl: Map of antenna/polarization visibility intputs to xGPU output indices.
        See optional sequence header entry ``ant_to_bl_id``. If not provided, this map
        should be available as a bifrost sequence header.
    :type antpol_to_bl: 4D list of int

    :param bl_is_conj: Map of visibility index to conjugation convention. See optional
        sequence header entry ``bl_is_conj``. If not provided, this map should be available
        as a bifrost sequence header.
    :type bl_is_conj: 4D list of bool

    :param checkfile: Path to a data file containing the expected correlator output. If provided,
        the data input to this block will be checked against this file. The data file should
        contain binary numpy.complex-format data in order ``time x nchan x nstand x nstand x npol x npol``.
        Each entry in this data file should represent an expected visibility which has been integrated
        for ``checkfile_acc_len`` samples. This file can be generated with this package's
        ``make_golden_inputs.py`` script.
    :type checkfile: str

    :param checkfile_acc_len: The number of integrations which have gone into each time slice of the
        provided ``checkfile``. For a check to be run, the accumulation length (and accumulation starts)
        of data input to this block should be a multiple of ``checkfile_acc_len``.
    :type checkfile_acc_len: int

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
        | ``dest_file``    | string |         | If not `""`, overrides       |
        |                  |        |         | ``dest_ip`` and causes the   |
        |                  |        |         | output data to be written to |
        |                  |        |         | the supplied file            |
        +------------------+--------+---------+------------------------------+
        | ``dest_port``    | int    |         | UDP port to which packets    |
        |                  |        |         | should be transmitted.       |
        +------------------+--------+---------+------------------------------+
        | ``max_mbps``     | int    | Mbits/s | The maximum output data rate |
        |                  |        |         | to allow before throttling.  |
        |                  |        |         | Set to ``-1`` to send as     |
        |                  |        |         | fast as possible.            |
        +------------------+--------+---------+------------------------------+


    **Output Data Format**

    Each packet from the correlator contains data from multiple channels for a single,
    dual-polarization baseline. There are two possible output formats depending on the
    value of ``use_cor_fmt`` with which this block is instantiated.

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
        | frame_number  | uint24     |                | Mark 5C frame number. Unused.               |
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
        |               |            |                | The second axis is the polatizaion of the   |
        |               |            |                | antenna at stand_i. The second axis is the  |
        |               |            |                | polarization of the antenna at stand_j.|    |
        |               |            |                | The fourth axis is complexity, with index 0 |
        |               |            |                | the real part of the visibility, and index  |
        |               |            |                | 1 the imaginary part.                       |
        +---------------+------------+----------------+---------------------------------------------+

    If ``use_cor_fmt=False``, this block outputs a stream of UDP packets, with each comprising
    a 56 byte header followed by a payload of signed 32-bit integers. The packet definition is
    as follows:

    .. code:: C
    
          struct corr_output_full_packet {
            uint64_t  sync_time;
            uint64_t  spectra_id;
            double    bw_hz;
            double    sfreq_hz;
            uint32_t  acc_len;
            uint32_t  nchans;
            uint32_t  chan0;
            uint32_t  npols;
            uint32_t  stand0;
            uint32_t  stand1;
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
        | spectra\_id   | int                 |                | The spectrum number for the first spectra   |
        |               |                     |                | which contributed to this packet’s          |
        |               |                     |                | integration.                                |
        +---------------+---------------------+----------------+---------------------------------------------+
        | bw\_hz        | double (binary64)   | Hz             | The total bandwidth of data in this packet  |
        +---------------+---------------------+----------------+---------------------------------------------+
        | sfreq\_hz     | double (binary64)   | Hz             | The center frequency of the first channel   |
        |               |                     |                | of data in this packet                      |
        +---------------+---------------------+----------------+---------------------------------------------+
        | acc\_len      | uint32              |                | The number of spectra integrated in this    |
        |               |                     |                | packet                                      |
        +---------------+---------------------+----------------+---------------------------------------------+
        | nchans        | uint32              |                | The number of frequency channels in this    |
        |               |                     |                | packet. For LWA-352 this is 184             |
        +---------------+---------------------+----------------+---------------------------------------------+
        | chan0         | uint32              |                | The index of the first frequency channel in |
        |               |                     |                | this packet                                 |
        +---------------+---------------------+----------------+---------------------------------------------+
        | npols         | uint32              |                | The number of polarizations of data in this |
        |               |                     |                | packet. For LWA-352, this is 2.             |
        +---------------+---------------------+----------------+---------------------------------------------+
        | stand0        | uint32              |                | The index of the first antenna stand in     |
        |               |                     |                | this packet’s visibility.                   |
        +---------------+---------------------+----------------+---------------------------------------------+
        | stand1        | uint32              |                | The index of the second antenna stand in    |
        |               |                     |                | this packet’s visibility.                   |
        +---------------+---------------------+----------------+---------------------------------------------+
        | data          | int32\*             |                | The data payload. Data for the visibility   |
        |               |                     |                | of antennas at stand0 and stand1, with      |
        |               |                     |                | stand1 conjugated. Data are a               |
        |               |                     |                | multidimensional array of 32-bit integers,  |
        |               |                     |                | with dimensions [``npols``, ``npols``,      |
        |               |                     |                | ``nchans``, 2]. The first axis is the       |
        |               |                     |                | polarization of the antenna at stand0. The  |
        |               |                     |                | second axis is the polarization of the      |
        |               |                     |                | antenna at stand1. The third axis is        |
        |               |                     |                | frequency channel. The fourth axis is       |
        |               |                     |                | complexity, with index 0 the real part of   |
        |               |                     |                | the visibility, and index 1 the imaginary   |
        |               |                     |                | part.                                       |
        +---------------+---------------------+----------------+---------------------------------------------+



    """
    def __init__(self, log, iring,
                 guarantee=True, core=-1, nchan=192, npol=2, nstand=352, etcd_client=None, dest_port=10000,
                 checkfile=None, checkfile_acc_len=1, antpol_to_bl=None, bl_is_conj=None, use_cor_fmt=True):
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        super(CorrOutputFull, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        self.nchan = nchan
        self.npol = npol
        self.nstand = nstand
        self.matlen = nchan * (nstand//2+1)*(nstand//4)*npol*npol*4

        self.igulp_size = self.matlen * 8 # complex64

        # Arrays to hold the conjugation and bl indices of data coming from xGPU
        self.antpol_to_bl = BFArray(np.zeros([nstand, nstand, npol, npol], dtype=np.int32), space='system')
        self.bl_is_conj   = BFArray(np.zeros([nstand, nstand, npol, npol], dtype=np.int32), space='system')
        if antpol_to_bl is not None:
            self.antpol_to_bl[...] = antpol_to_bl
        if bl_is_conj is not None:
            self.bl_is_conj[...] = bl_is_conj
        self.reordered_data = BFArray(np.zeros([nstand, nstand, npol, npol, nchan, 2], dtype=np.int32), space='system')
        self.dump_size = nstand * (nstand+1) * npol * npol * nchan * 2 * 4 / 2.

        self.checkfile_acc_len = checkfile_acc_len
        if checkfile is None:
            self.checkfile = None
        else:
            self.checkfile = open(checkfile, 'rb')
            self.checkfile_nbytes = os.path.getsize(checkfile)
            self.log.info("CORR OUTPUT >> Checkfile %s" % self.checkfile.name)
            self.log.info("CORR OUTPUT >> Checkfile length: %d bytes" % self.checkfile_nbytes)
            self.log.info("CORR OUTPUT >> Checkfile accumulation length: %d" % self.checkfile_acc_len)
        self.use_cor_fmt = use_cor_fmt
        self.output_file = None
        if self.use_cor_fmt:
            self.sock = None
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(0)

        self.define_command_key('dest_ip', type=str, initial_val='0.0.0.0')
        self.define_command_key('dest_file', type=str, initial_val='')
        self.define_command_key('dest_port', type=int, initial_val=dest_port)
        self.define_command_key('max_mbps', type=int, initial_val=-1)
        self.update_command_vals()

    def get_checkfile_corr(self, t):
        """
        Get a single integration from the test file,
        looping back to the beginning of the file when
        the end is reached.
        Inputs: t (int) -- time index of correlation
        """
        dim = np.array([self.nchan, self.nstand, self.nstand, self.npol, self.npol])
        nbytes = dim.prod() * 2 * 8
        seekloc = (nbytes * t) % self.checkfile_nbytes
        self.log.debug("CORR OUTPUT >> Testfile has %d bytes. Seeking to %d and reading %d bytes for sample %d" % (self.checkfile_nbytes, seekloc, nbytes, t))
        self.checkfile.seek(seekloc)
        dtest_raw = self.checkfile.read(nbytes)
        if len(dtest_raw) != nbytes:
            self.log.error("CORR OUTPUT >> Failed to get correlation matrix from checkfile")
            return np.zeros(dim, dtype=np.complex)
        return np.frombuffer(dtest_raw, dtype=np.complex).reshape(dim)

    def send_packets_py(self, sync_time, this_gulp_time, bw_hz, sfreq,
                              upstream_acc_len, chan0):
        cpu_affinity.set_core(self.core)
        start_time = time.time()
        packet_cnt = 0
        header_static = struct.pack(">QQ2d4I",
                                    sync_time,
                                    this_gulp_time,
                                    bw_hz,
                                    sfreq,
                                    upstream_acc_len,
                                    self.nchan,
                                    chan0,
                                    self.npol,
                                    )
        pkt_payload_bits = self.nchan * self.npol * self.npol * 8 * 8
        block_bits_sent = 0
        start_time = time.time()
        block_start = time.time()
        for s0 in range(self.nstand):
            for s1 in range(s0, self.nstand):
                header_dyn = struct.pack(">2I", s0, s1)
                self.sock.sendto(header_static + header_dyn + self.reordered_data[s0, s1].tobytes(), (self.command_vals['dest_ip'], self.command_vals['dest_port']))
                if self.command_vals['max_mbps'] > 0:
                    block_bits_sent += pkt_payload_bits
                    # Apply throttle every 1MByte -- every >~100 packets
                    if block_bits_sent > 8000000:
                        block_elapsed = time.time() - block_start
                        # Minimum allowed time to satisfy max rate
                        min_time = block_bits_sent / (1.e6 * self.command_vals['max_mbps'])
                        delay = min_time - block_elapsed
                        if delay > 0:
                            time.sleep(delay)
                        block_start = time.time()
                        block_bits_sent = 0
        stop_time = time.time()
        elapsed = stop_time - start_time
        gbps = 8*self.dump_size / elapsed / 1e9
        self.log.info("CORR OUTPUT >> Sending complete for time %d in %.2f seconds (%f Gb/s)" % (this_gulp_time, elapsed, gbps))
        self.update_stats({'output_gbps': gbps})

    def send_packets_bf(self, udt, this_gulp_time, desc, chan0, gain, navg, tuning):
        cpu_affinity.set_core(self.core)
        start_time = time.time()
        desc.set_chan0(chan0)
        desc.set_gain(gain)
        desc.set_decimation(navg)
        desc.set_nsrc((self.nstand*(self.nstand + 1))//2)
        desc.set_tuning(tuning)
        pkt_payload_bits = self.nchan * self.npol * self.npol * 8 * 8
        block_bits_sent = 0
        start_time = time.time()
        block_start = time.time()
        for i in range(self.nstand):
            # `data` should be in order stand1 x stand0 x chan x pol1 x pol0
            # copy a single baseline of data
            sdata = self.reordered_data[i, i:, :, :].copy(space='system').view('cf32') # packet format expects floats
            # reshape and send
            sdata = sdata.reshape(1, -1, self.nchan*self.npol*self.npol).view('cf32')
            udt.send(desc, this_gulp_time, 0, i*(2*self.nstand + 1 - i) // 2 + i, 1, sdata)      
            if self.command_vals['max_mbps'] > 0:
                block_bits_sent += i * self.nchan * self.npol * self.npol * 8 * 8
                # Apply throttle every 1MByte -- every >~100 packets
                if block_bits_sent > 8000000:
                    block_elapsed = time.time() - block_start
                    # Minimum allowed time to satisfy max rate
                    min_time = block_bits_sent / (1.e6 * self.command_vals['max_mbps'])
                    delay = min_time - block_elapsed
                    if delay > 0:
                        time.sleep(delay)
                    block_start = time.time()
                    block_bits_sent = 0
        del sdata
        stop_time = time.time()
        elapsed = stop_time - start_time
        gbps = 8 * self.dump_size / elapsed / 1e9
        self.log.info("CORR OUTPUT >> Sending complete for time %d in %.2f seconds (%f Gb/s)" % (this_gulp_time, elapsed, gbps))
        self.update_stats({'output_gbps': gbps})

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        prev_time = time.time()
        self.sock = UDPSocket()
        self.sock.connect(Address('10.41.0.19', 10010))
        udt = UDPTransmit('cor_%i' % self.nchan, sock=self.sock, core=self.core)
        for iseq in self.iring.read(guarantee=self.guarantee):
            self.update_pending = True # Reprocess commands on each new sequence
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            upstream_acc_len = ihdr['acc_len']
            upstream_start_time = this_gulp_time
            nchan = ihdr['nchan']
            system_nchan = ihdr['system_nchan']
            chan0 = ihdr['chan0']
            bw_hz = ihdr['bw_hz']
            sfreq = ihdr['sfreq']
            npol  = ihdr['npol']
            npipeline = system_nchan // nchan
            this_pipeline = (chan0 // nchan) % npipeline
            if 'ant_to_bl_id' in ihdr:
                self.antpol_to_bl[...] = ihdr['ant_to_bl_id']
            if 'bl_is_conj' in ihdr:
                self.bl_is_conj[...] = ihdr['bl_is_conj']
            for ispan in iseq.read(self.igulp_size):
                if ispan.size < self.igulp_size:
                    continue # skip last gulp
                # Update destinations if necessary
                if self.update_pending:
                    self.update_command_vals()
                    if self.command_vals['dest_file'] != "":
                        self.log.info("CORR OUTPUT >> Updating destination to file %s (max Mbps %.1f ns)" % 
                                      (self.command_vals['dest_file'], self.command_vals['max_mbps']))
                    else:
                        self.log.info("CORR OUTPUT >> Updating destination to %s:%s (max Mbps %.1f ns)" % 
                                      (self.command_vals['dest_ip'], self.command_vals['dest_port'], self.command_vals['max_mbps']))
                    if self.use_cor_fmt:
                        #if self.sock:
                        #    del self.sock
                        #    self.sock = None
                        #if udt:
                        #    del udt
                        #    udt = None
                        #if self.output_file: self.output_file.close()
                        #if self.command_vals['dest_file'] != "":
                        #    try:
                        #         filename = self.command_vals['dest_file']
                        #         self.log.info("CORR OUTPUT >> Trying to open file %s for output" % filename)
                        #         self.output_file = open(filename, "wb")
                        #    except:
                        #         self.log.error("CORR OUTPUT >> Tried to open file %s for output but failed" % filename)
                        #    self.sock = None
                        #    udt = DiskWriter('cor_%i' % self.nchan, self.output_file, core=self.core)
                        #else:
                        #    self.sock = UDPSocket()
                        #    self.sock.connect(Address(self.command_vals['dest_ip'], self.command_vals['dest_port']))
                        #    udt = UDPTransmit('cor_%i' % self.nchan, sock=self.sock, core=self.core)
                        desc = HeaderInfo()
                self.update_stats({'curr_sample':this_gulp_time})
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                _bf.bfXgpuReorder(ispan.data.as_BFarray(), self.reordered_data.as_BFarray(), self.antpol_to_bl.as_BFarray(), self.bl_is_conj.as_BFarray())
                # Check against test data if a file is provided
                if self.checkfile:
                    assert upstream_acc_len % self.checkfile_acc_len == 0, "CORR OUTPUT >> Testfile acc len not compatible with pipeline acc len"
                    assert upstream_start_time % self.checkfile_acc_len == 0, "CORR OUTPUT >> Testfile acc len not compatible with pipeline start time"
                    nblocks = (upstream_acc_len // self.checkfile_acc_len)
                    self.log.info("CORR OUTPUT >> Computing expected output from test file")
                    self.log.info("CORR OUTPUT >> Upstream accumulation %d" % upstream_acc_len)
                    self.log.info("CORR OUTPUT >> File accumulation %d" % self.checkfile_acc_len)
                    self.log.info("CORR OUTPUT >> Integrating %d blocks" % nblocks)
                    dtest = np.zeros([self.nchan, self.nstand, self.nstand, self.npol, self.npol], dtype=np.complex)
                    for i in range(nblocks):
                        dtest += self.get_checkfile_corr(this_gulp_time // self.checkfile_acc_len + i)
                    # check baseline by baseline
                    badcnt = 0
                    goodcnt = 0
                    nonzerocnt = 0
                    zerocnt = 0
                    now = time.time()
                    for s0 in range(self.nstand):
                        if time.time() - now > 15:
                            self.log.info("CORR OUTPUT >> Check complete for stand %d" % s0)
                            now = time.time()
                        for s1 in range(s0, self.nstand):
                            for p0 in range(self.npol):
                               for p1 in range(self.npol):
                                   if not np.all(self.reordered_data[s0, s1, p0, p1, :, 0] == 0):
                                       nonzerocnt += 1
                                   else:
                                       zerocnt += 1
                                   if not np.all(self.reordered_data[s0, s1, p0, p1, :, 1] == 0):
                                       nonzerocnt += 1
                                   else:
                                       zerocnt += 1
                                   if np.any(self.reordered_data[s0, s1, p0, p1, :, 0] != dtest[:, s0, s1, p0, p1].real):
                                       self.log.error("CORR OUTPUT >> test vector mismatch! [%d, %d, %d, %d] real" %(s0,s1,p0,p1))
                                       print("antpol to bl: %d" % self.antpol_to_bl[s0,s1,p0,p1])
                                       print("is conjugated : %d" % self.bl_is_conj[s0,s1,p0,p1])
                                       print("pipeline:", self.reordered_data[s0, s1, p0, p1, 0:5, 0])
                                       print("expected:", dtest[0:5, s0, s1, p0, p1].real)
                                       badcnt += 1
                                   else:
                                       goodcnt += 1
                                   if np.any(self.reordered_data[s0, s1, p0, p1, :, 1] != dtest[:, s0, s1, p0, p1].imag): # test data follows inverse conj convention
                                       self.log.error("CORR OUTPUT >> test vector mismatch! [%d, %d, %d, %d] imag" %(s0,s1,p0,p1))
                                       print("antpol to bl: %d" % self.antpol_to_bl[s0,s1,p0,p1])
                                       print("is conjugated : %d" % self.bl_is_conj[s0,s1,p0,p1])
                                       print("pipeline:", self.reordered_data[s0, s1, p0, p1, 0:5, 1])
                                       print("expected:", dtest[0:5, s0, s1, p0, p1].imag)
                                       badcnt += 1
                                   else:
                                       goodcnt += 1
                    if badcnt > 0:
                        self.log.error("CORR OUTPUT >> test vector check complete. Good: %d, Bad: %d, Non-zero: %d, Zero: %d" % (goodcnt, badcnt, nonzerocnt, zerocnt))
                    else:
                        self.log.info("CORR OUTPUT >> test vector check complete. Good: %d, Bad: %d, Non-zero: %d, Zero: %d" % (goodcnt, badcnt, nonzerocnt, zerocnt))

                if self.command_vals['dest_ip'] != "0.0.0.0" or self.command_vals['dest_file'] != "":
                    if self.use_cor_fmt:
                        self.send_packets_bf(udt, this_gulp_time, desc, chan0, 0, upstream_acc_len, (npipeline << 16) + (this_pipeline+1))
                    else:
                        self.send_packets_py(ihdr['sync_time'], this_gulp_time, bw_hz, sfreq, upstream_acc_len, chan0)
                else:
                    self.log.info("CORR OUTPUT >> Skipping sending for time %d" % this_gulp_time)
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.update_stats()
                # And, update overall time counter
                this_gulp_time += upstream_acc_len
        if self.checkfile:
            self.checkfile.close()
