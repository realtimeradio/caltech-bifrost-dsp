import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap
from bifrost.ndarray import copy_array
from bifrost.device import stream_synchronize, set_device as BFSetGPU
from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_writer import HeaderInfo, DiskWriter, UDPTransmit

import os
import time
import ujson as json
import socket
import struct
import numpy as np
from threading import Lock

from .block_base import Block

class BeamformOutput(Block):
    """
    **Functionality**

    This block reads beamformed power data from a CPU-side buffer and
    transmits as a stream of UDP packets.

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
        | ``nchan``     | int    |       | The number of frequency channels in the input  |
        |               |        |       | data buffer                                    |
        +---------------+--------+-------+------------------------------------------------+
        | ``chan0``     | int    |       | The index of the first frequency channel in    |
        |               |        |       | the input data buffer                          |
        +---------------+--------+-------+------------------------------------------------+
        | ``nbeam``     | int    |       | The number of beams in the input data buffer   |
        +---------------+--------+-------+------------------------------------------------+
        | ``nbit``      | int    |       | The number of bits (per real/imag part) of the |
        |               |        |       | input data samples. Must be 32                 |
        +---------------+--------+-------+------------------------------------------------+
        | ``complex``   | Bool   |       | True indicates that the input samples are      |
        |               |        |       | complex, with real and imaginary parts both    |
        |               |        |       | having ``nbit`` bits. Must be True.            |
        +---------------+--------+-------+------------------------------------------------+
        | ``system_ncha | int    |       | The total number of frequency channels in the  |
        | n``           |        |       | multi-pipeline system. Must be a multiple of   |
        |               |        |       | ``nchan``.                                     |
        +---------------+--------+-------+------------------------------------------------+
        | ``npol``      | int    |       | The number of polarizations in the input data  |
        |               |        |       | buffer. Must be 1                              |
        +---------------+--------+-------+------------------------------------------------+
        | ``fs_hz``     | int    | Hz    | ADC sample rate.                               |
        +---------------+--------+-------+------------------------------------------------+
        | ``bw_hz``     | double | Hz    | Bandwidth of the input data.                   |
        +---------------+--------+-------+------------------------------------------------+

    **Output Headers**

    This is a bifrost sink block, and provides no data to an output ring.

    **Data Buffers**

    *Input Data Buffer*: A CPU-side bifrost ring buffer of 32 bit,
    floating-point, integrated, beam powers.
    Data has dimensionality ``beams x time x channel x beam-element``.

    ``channel`` runs from 0 to ``nchan``.
    
    ``beam`` runs from 0 to the output ``nbeam-1`` (equivalent to the input ``nbeam/2 - 1``).
    
    ``beam-element`` runs from 0 to 3 with the following mapping:

      - index 0: The accumulated power of a beam's ``X`` polarization
      - index 1: The accumulated power of a beam's ``Y`` polarization
      - index 2: The accumulated real part of a beam's ``X x conj(Y)`` cross-power
      - index 3: The accumulated imaginary part of a beam's ``X * conj(Y)`` cross-power

    
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

    :param ntime_gulp: Number of time samples to copy with each gulp.
    :type ntime_gulp: int

    :param dest_port: Default destination port for UDP data. Can be overriden with the runtime
        control interface.
    :type dest_port: int

    :param nchan: Number of frequency channels processed per beam.
    :type nchan: int

    :param nbeam: Number of beams processed.
    :type nbeam: int

    **Runtime Control and Monitoring**

    This block reads the following control keys

    .. table::
        :widths: 25 10 10 55

        +------------------+-------------+---------+------------------------------+
        | Field            | Format      | Units   | Description                  |
        +==================+=============+=========+==============================+
        | ``dest_ip``      | list of     |         | Destination IP addresses for |
        |                  | string      |         | transmitted packets, in      |
        |                  |             |         | dotted-quad format. Eg.      |
        |                  |             |         | ``"10.0.0.1"``. Use          |
        |                  |             |         | ``"0.0.0.0"`` to skip        |
        |                  |             |         | sending packets. Beam ``i``  |
        |                  |             |         | is sent to ``dest_ip[i %     |
        |                  |             |         | len(dest_ip)]``.             |
        +------------------+-------------+---------+------------------------------+
        | ``dest_port``    | list of int |         | UDP port to which packets    |
        |                  |             |         | should be transmitted. Beam  |
        |                  |             |         | ``i`` is sent to             |
        |                  |             |         | ``dest_port[i] %             |
        |                  |             |         | len(dest_port)``             |
        +------------------+-------------+---------+------------------------------+

    **Output Data Format**

    Each packet output contains a single time sample of data from multiple channels
    and a single power beam.
    The output data format complies with bifrost's built-in "PBEAM" spec.
   
    This format comprises
    a stream of UDP packets, each with a 18 byte header defined as follows:

    .. code:: C
    
          struct ibeam {
              uint8_t  server; // 1-indexed
              uint8_t  beam;   // 1-indexed
              uint8_t  gbe;    // AKA "tuning"
              uint8_t  nchan;
              uint8_t  nbeam;
              uint8_t  nserver;
              uint16_t navg;   // Number of raw spectra averaged
              uint16_t chan0;  // First channel index in a packet
              uint64_t seq;    // 1-indexed: Really?
              float    data[nchan, nbeam, 4]; // Channel x Beam x Beam Element x 32-bit float
          };

    Packet fields are as follows:

    .. table::
        :widths: 25 10 10 55

        +---------------+------------+--------+---------------------------------------------+
        | Field         | Format     | Units  | Description                                 |
        +===============+============+========+=============================================+
        | server        | uint8      |        | One-based "pipeline number". Pipeline 1     |
        |               |            |        | processes the first ``nchan`` channels,     |
        |               |            |        | pipeline ``p`` processes the ``p``-th       |
        |               |            |        | ``nchan`` channels. Pipline ID counts       |
        |               |            |        | through each channel block, and then        |
        |               |            |        | multiple beams in the system. Eg, if the    |
        |               |            |        | system has 512 channels, 256 channels per   |
        |               |            |        | packet, and 3 beams, ``server`` runs from 1 |
        |               |            |        | to 6.                                       |
        +---------------+------------+--------+---------------------------------------------+
        | beam          | uint8      |        | One-based "beam number".                    |
        +---------------+------------+--------+---------------------------------------------+
        | gbe           | uint8      |        | AKA "tuning". Set to 0.                     |
        +---------------+------------+--------+---------------------------------------------+
        | nchan         | uint8      |        | Number of frequency channels in this packet |
        +---------------+------------+--------+---------------------------------------------+
        | nbeam         | uint8      |        | Number of beams in this packet. Currently   |
        |               |            |        | always 1.                                   |
        +---------------+------------+--------+---------------------------------------------+
        | nserver       | uint8      |        | The total number of pipelines in the system |
        |               |            |        | times the number of beams per pipeline      |
        +---------------+------------+--------+---------------------------------------------+
        | chan0         | uint32     |        | Zero-indexed ID of the first frequency      |
        |               |            |        | channel in this packet.                     |
        +---------------+------------+--------+---------------------------------------------+
        | seq           | uint64     | ADC    | Central sampling time since 1970-01-01      |
        |               |            | sample | 00:00:00 UTC.                               |
        |               |            | period |                                             |
        +---------------+------------+--------+---------------------------------------------+
        | data          | float      |        | Data payload. Beam powers, in order         |
        |               |            |        | (slowest to fastest) ``Channel x Beam x     |
        |               |            |        | Beam Element``. Beam elements are ``[XX,    |
        |               |            |        | YY, real(XY), imag(XY)]``. Data are sent in |
        |               |            |        | native host endianness                      |
        +---------------+------------+--------+---------------------------------------------+
 
    """

    def __init__(self, log, iring,
                 guarantee=True, core=-1, etcd_client=None, dest_port=10000,
                 ntime_gulp=480, pipeline_idx=1, nchan=96, nbeam=16,
                 ):
        super(BeamformOutput, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)

        self.ntime_gulp = ntime_gulp
        self.pipeline_idx = pipeline_idx
        self.nchan = nchan
        self.nbeam = nbeam
        self._etcd_sets_pending = False # Manage the update_pending flag ourselves
        self.define_command_key('dest_ip', type=list, initial_val=['0.0.0.0'])
        self.define_command_key('dest_port', type=list, initial_val=[dest_port])
        self.update_command_vals()
        self.use_python_tx = False # Python socket output is not tested
        # Populate the initial IP / Port / Sockets / UDTs
        self.udts       = [None for _ in range(self.nbeam)]
        self.socks      = [None for _ in range(self.nbeam)]
        self.beam_ips   = [None for _ in range(self.nbeam)]
        self.beam_ports = [None for _ in range(self.nbeam)]
        for beam in range(self.nbeam):
            self.beam_ips[beam] = self.command_vals['dest_ip'][beam % len(self.command_vals['dest_ip'])]
            self.beam_ports[beam] = self.command_vals['dest_port'][beam % len(self.command_vals['dest_port'])]
        self.tx_locks = [Lock() for _ in range(self.nbeam)]

    def _etcd_callback(self, watchresponse):
        """
        A callback executed whenever this block's command key is modified.

        This callback JSON decodes the key contents, and passes the
        resulting dictionary to ``_process_commands``.
        The ``last_cmd_response`` status value is set to the return value of
        ``_process_commands`` to indicate any error conditions

        :param watchresponse: A WatchResponse object used by the etcd
            `add_watch_prefix_callback` as the calling argument.
        :type watchresponse: WatchResponse
        """
        cpu_affinity.set_core(self.core)
        # Empirically determined sleep patterns.
        # Destroying and creating UDTs is liable to lock up bifrost momentarily
        SLEEP_TIME = 0.1 # sleep seconds before and after UDT creation
        SLEEP_TIME_POST_DEL = 0.3 # sleep seconds after destroying an existing UDT
        super(BeamformOutput, self)._etcd_callback(watchresponse)
        self.update_command_vals()

        t0 = time.time()
        for beam in range(self.nbeam):
            ip = self.command_vals['dest_ip'][beam % len(self.command_vals['dest_ip'])]
            port = self.command_vals['dest_port'][beam % len(self.command_vals['dest_port'])]
            if self.beam_ips[beam] == ip and self.beam_ports[beam] == port:
                continue
            self.tx_locks[beam].acquire()
            time.sleep(SLEEP_TIME)
            if self.udts[beam] is not None:
                self.udts[beam] = None
                time.sleep(SLEEP_TIME_POST_DEL)
            self.beam_ips[beam] = ip
            self.beam_ports[beam] = port
            if self.socks[beam] is not None:
                self.socks[beam].close()
            self.log.info("Sending beam %d to %s:%d" % (beam, ip, port))
            if not self.use_python_tx:
                self.socks[beam] = UDPSocket()
                self.socks[beam].connect(Address(ip, port))
                self.udts[beam] = UDPTransmit('pbeam1_%d' % (self.nchan), sock=self.socks[beam], core=self.core)
            else:
                self.socks[beam] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socks[beam].connect((ip, port))
            time.sleep(SLEEP_TIME)
            self.tx_locks[beam].release()
        self.stats.update({'dest_ip': self.beam_ips,
            'dest_port': self.beam_ports,
            'update_pending': False,
            'last_update_time': time.time()})
        self.update_stats()
        t1 = time.time()
        self.log.info("Returning after %f secs" % (t1-t0))

    def send_packets_python(self, src, tuning, nsrc, navg, chan0, seq, b, d):
        header0 = np.array([src, b, tuning, self.nchan, self.nbeam, nsrc], dtype='>u1').tobytes() \
                + np.array([navg, chan0], dtype='>u2').tobytes()
        header1 = np.array([seq], dtype='>u8')
        for t in range(self.ntime_gulp):
            self.socks[b].send(header0 + header1.tobytes() + d[t].tobytes())
            header1[0] += self.ntime_gulp * navg

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        prev_time = time.time()
        socks = None
        udts = None
        desc = None
        for iseq in self.iring.read(guarantee=self.guarantee):
            # Update control each sequence
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            upstream_acc_len = ihdr['acc_len']
            assert self.nchan == ihdr['nchan']
            assert self.nbeam == ihdr['nbeam']
            nbit  = ihdr['nbit']
            system_nchan = ihdr['system_nchan']
            npipeline = system_nchan // self.nchan
            chan0 = ihdr['chan0']
            npol  = ihdr['npol']
            samples_per_spectra = int(self.nchan * ihdr['fs_hz'] / ihdr['bw_hz'])
            this_pipeline = (chan0 // self.nchan) % npipeline
            igulp_size = self.ntime_gulp * self.nchan * self.nbeam * npol**2 * nbit // 8
            packet_cnt = 0
            if desc: del desc
            desc = HeaderInfo()
            desc.set_chan0(chan0)
            desc.set_tuning(1)
            desc.set_nchan(self.nchan)
            desc.set_decimation(upstream_acc_len) # Sets navg field
            desc.set_nsrc(npipeline)
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # ignore final gulp
                self.stats['curr_sample'] = this_gulp_time
                self.update_stats()
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                idata = ispan.data.view('f32').reshape([self.nbeam, self.ntime_gulp, self.nchan, npol**2])
                start_time = time.time()
                time_tag = this_gulp_time * samples_per_spectra
                for beam in range(self.nbeam):
                    if not self.tx_locks[beam].acquire(False):
                        continue
                    if self.beam_ips[beam] != '0.0.0.0':
                        idata_beam = idata[beam,:,:,:].reshape(self.ntime_gulp, 1, self.nchan * npol**2)
                        try:
                            if self.use_python_tx:
                                self.send_packets_python(self.pipeline_idx-1, 1, npipeline,
                                        upstream_acc_len, chan0, this_gulp_time, beam, idata_beam)
                            else:
                                self.udts[beam].send(desc, this_gulp_time, upstream_acc_len,
                                            self.pipeline_idx-1, 0, idata_beam)
                        except Exception as e:
                            self.log.error("BEAM OUTPUT >> Sending error (beam %d): %s" % (beam, str(e)))
                    self.tx_locks[beam].release()
                stop_time = time.time()
                elapsed = stop_time - start_time
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.stats['last_end_sample'] = this_gulp_time
                self.update_stats()
                # And, update overall time counter
                this_gulp_time += upstream_acc_len * self.ntime_gulp
