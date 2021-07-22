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
        | ``fs_hz``     | int    | Hz    | ADC sample rate.
        +---------------+--------+-------+------------------------------------------------+

    **Output Headers**

    This is a bifrost sink block, and provides no data to an output ring.

    **Data Buffers**

    *Input Data Buffer*: A CPU-side bifrost ring buffer of 32 bit,
    floating-point, integrated, beam powers.
    Data has dimensionality ``time x channel x beams x beam-element``.

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
        | seq           | uint64     | ADC sample     | Central sampling time since 1970-01-01      |
        |               |            | period         | 00:00:00 UTC.                               |
        +---------------+------------+----------------+---------------------------------------------+
        | data          | float      |        | Data payload. Beam powers, in order         |
        |               |            |        | (slowest to fastest) ``Channel x Beam x     |
        |               |            |        | Beam Element``. Beam elements are ``[XX,    |
        |               |            |        | YY, real(XY), imag(XY)]``. Data are sent in |
        |               |            |        | native host endianness                      |
        +---------------+------------+--------+---------------------------------------------+
 
    """

    def __init__(self, log, iring,
                 guarantee=True, core=-1, etcd_client=None, dest_port=10000,
                 ntime_gulp=480,
                 ):
        super(BeamformOutput, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)

        self.ntime_gulp = ntime_gulp
        self.define_command_key('dest_ip', type=list, initial_val=['0.0.0.0'])
        self.define_command_key('dest_port', type=list, initial_val=[dest_port])
        self.update_command_vals()

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
            self.update_pending = True
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            upstream_acc_len = ihdr['acc_len']
            nchan = ihdr['nchan']
            nbeam = ihdr['nbeam']
            nbit  = ihdr['nbit']
            system_nchan = ihdr['system_nchan']
            npipeline = system_nchan // nchan
            chan0 = ihdr['chan0']
            npol  = ihdr['npol']
            samples_per_spectra = int(nchan_sum * nchan * ihdr['fs_hz'] / bw_hz)
            this_pipeline = (chan0 // nchan) % npipeline
            igulp_size = self.ntime_gulp * nchan * nbeam * npol**2 * nbit // 8
            packet_cnt = 0
            if desc: del desc
            if socks: del socks
            if udts: del udts
            desc = HeaderInfo()
            socks = [UDPSocket() for beam in range(nbeam)]
            # Seem to need to connect the sockets to _something_ prior
            # to handing them to UDPTransmit. Can update their destinations
            # later.
            for sock in socks:
                sock.connect(Address('0.0.0.0', 60000))
            udts  = [UDPTransmit('pbeam1_%d' % (nchan), sock=socks[beam], core=self.core) for beam in range(nbeam)]
            beam_ips = ['0.0.0.0' for beam in range(nbeam)]
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # ignore final gulp
                # Update destinations if necessary
                if self.update_pending:
                    self.update_command_vals()
                    self.log.info("BEAM OUTPUT >> Updating destination to %s:%s" 
                        % (self.command_vals['dest_ip'], self.command_vals['dest_port']))
                    for beam in range(nbeam):
                        ip = self.command_vals['dest_ip'][beam % len(self.command_vals['dest_ip'])]
                        port = self.command_vals['dest_port'][beam % len(self.command_vals['dest_port'])]
                        beam_ips[beam] = ip # Needed to use '0.0.0.0' as per-beam disable
                        if ip != '0.0.0.0':
                            self.log.info("BEAM OUTPUT >> Will send beam %d to %s:%d" % (beam, ip, port))
                        socks[beam].close()
                        socks[beam].connect(Address(ip, port))
                    desc.set_chan0(chan0)
                    desc.set_tuning(0)
                    desc.set_nchan(nchan)
                    desc.set_decimation(upstream_acc_len) # Sets navg field
                    desc.set_nsrc(nbeam * npipeline)
                    self.stats.update({'dest_ip': self.command_vals['dest_ip'],
                                       'dest_port': self.command_vals['dest_port'],
                                       'update_pending': self.update_pending,
                                       'last_update_time': time.time()})
                self.stats['curr_sample'] = this_gulp_time
                self.update_stats()
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                idata = ispan.data.view('f32').reshape([self.ntime_gulp, nbeam, nchan, npol**2])
                start_time = time.time()
                time_tag = this_gulp_time * samples_per_spectra
                for beam in range(nbeam):
                    if beam_ips[beam] != '0.0.0.0':
                        idata_beam = idata[:,beam,:,:].copy('system').reshape(self.ntime_gulp, 1, nchan * npol**2)
                        try:
                            udts[beam].send(desc, this_gulp_time, samples_per_spectra,
                                            npipeline*nbeam + this_pipeline, 0, idata_beam)
                        except Exception as e:
                            self.log.error("BEAM OUTPUT >> Sending error: %s" % (str(e)))
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
