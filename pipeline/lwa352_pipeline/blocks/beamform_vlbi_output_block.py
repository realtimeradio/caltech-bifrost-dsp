import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap
from bifrost.ndarray import copy_array
from bifrost.device import stream_synchronize, set_device as BFSetGPU
#from bifrost.transpose import transpose as Transpose
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

class BeamformVlbiOutput(Block):
    """
    **Functionality**

    Output a stream of UDP packets containing multiple beam voltages

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

    **Output Headers**

    This is a bifrost sink block, and provides no data to an output ring.

    **Data Buffers**

    *Input Data Buffer*: A CPU- or GPU-side bifrost ring buffer of 32+32 bit complex
    floating-point data containing beamformed voltages.
    Data have dimensions (slowest to fastest):
    ``channel x beams x time x complexity``. This buffer is read in blocks of
    ``ntime_gulp`` samples.

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

    :param ntime_gulp: Number of time samples to read on each loop iteration.
    :type ntime_gulp: int


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

    Each packet output contains a single time sample of data from multiple channels
    and multiple voltage beams.
    The output data format complies with the LWA-SV "IBEAM"
    spec This format comprises
    a stream of UDP packets, each with a 32 byte header defined as follows:

    .. code:: C
    
          struct ibeam {
              uint8_t  server;
              uint8_t  gbe;
              uint8_t  nchan;
              uint8_t  nbeam;
              uint8_t  nserver;
              uint16_t chan0;
              uint64_t seq;
              float    data[nchan, nbeam, 2]; // Channel x Beam x Complexity x 32-bit float
          };

    Packet fields are as follows:

    .. table::
        :widths: 25 10 10 55

        +---------------+------------+----------------+---------------------------------------------+
        | Field         | Format     | Units          | Description                                 |
        +===============+============+================+=============================================+
        | server        | uint8      |                | One-based "pipeline number". Pipeline 1     |
        |               |            |                | processes the first ``nchan`` channels,     |
        |               |            |                | pipeline ``p`` processes the ``p``-th       |
        |               |            |                | ``nchan`` channels.                         |
        +---------------+------------+----------------+---------------------------------------------+
        | gbe           | uint8      |                | AKA "tuning". Set to 0.                     |
        +---------------+------------+----------------+---------------------------------------------+
        | nchan         | uint8      |                | Number of frequency channels in this packet |
        +---------------+------------+----------------+---------------------------------------------+
        | nbeam         | uint8      |                | Number of beams in this packet              |
        +---------------+------------+----------------+---------------------------------------------+
        | nserver       | uint8      |                | The total number of pipelines in the        |
        |               |            |                | system.                                     |
        +---------------+------------+----------------+---------------------------------------------+
        | chan0         | uint32     |                | Zero-indexed ID of the first frequency      |
        |               |            |                | channel in this packet.                     |
        +---------------+------------+----------------+---------------------------------------------+
        | seq           | uint64     |                | Zero-indexed spectra number for the spectra |
        |               |            |                | in this packet. Specified relative to the   |
        |               |            |                | system synchronization time.                |
        +---------------+------------+----------------+---------------------------------------------+
        | data          | float      |                | Data payload. Beam voltages, in order       |
        |               |            |                | (slowest to fastest) ``Channel x Beam x     |
        |               |            |                | Complexity``                                |
        +---------------+------------+----------------+---------------------------------------------+

    """

    def __init__(self, log, iring,
                 guarantee=True, core=-1, etcd_client=None, dest_port=10000,
                 ntime_gulp=480, pipeline_idx=1
                 ):
        super(BeamformVlbiOutput, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        cpu_affinity.set_core(self.core)

        self.sock = None
        self.define_command_key('dest_ip', type=str, initial_val='0.0.0.0')
        self.define_command_key('dest_port', type=int, initial_val=dest_port)
        self.update_command_vals()
        self.dest_ip = self.command_vals['dest_ip']
        self.dest_port = self.command_vals['dest_port']
        self.ntime_gulp = ntime_gulp
        self._npacket_burst = 32 # Number of packets to burst between throttle sleep calls
        self._max_bps = 0.6 * 1e9
        self.pipeline_idx = pipeline_idx
        self.nbeam_send = 1
        self.npol = 2 # If the upstream beamformer provides single pol data, we interpret pairs of beams as dual-pol

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        udt = None
        prev_time = time.time()
        for iseq in self.iring.read(guarantee=self.guarantee):
            self.update_pending = True
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            this_gulp_time = ihdr['seq0']
            nchan = ihdr['nchan']
            nbeam = ihdr['nbeam']
            nbit  = ihdr['nbit']
            system_nchan = ihdr['system_nchan']
            chan0 = ihdr['chan0']
            npol  = ihdr['npol']
            igulp_size = self.ntime_gulp * nbeam * nchan * npol * 2 * nbit // 8
            
            nbeampol = self.nbeam_send*self.npol
            nbeamset = nbeam*npol // nbeampol
            
            packet_cnt = 0
            desc = HeaderInfo()
            desc.set_nchan(nchan)
            desc.set_chan0(chan0)
            desc.set_nsrc(system_nchan // nchan)
            desc.set_tuning(1)
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # ignore final gulp
                # Update destinations if necessary
                if self.update_pending:
                    self.update_command_vals()
                    # Don't do anything unless something has changed
                    if not (self.dest_ip == self.command_vals['dest_ip'] and self.dest_port == self.command_vals['dest_port']):
                        self.dest_ip = self.command_vals['dest_ip']
                        self.dest_port = self.command_vals['dest_port']
                        self.log.info("VLBI OUTPUT >> Updating destination to %s:%s" % 
                                (self.dest_ip, self.dest_port))
                        if self.sock is not None:
                            self.sock.close()
                        self.sock = UDPSocket()
                        self.sock.connect(Address(self.dest_ip, self.dest_port))
                        udt = UDPTransmit('ibeam%i_%i' % (self.nbeam_send, nchan), sock=self.sock, core=self.core)
                    self.stats.update({'dest_ip': self.dest_ip,
                                       'dest_port': self.dest_port,
                                       'update_pending': self.update_pending,
                                       'last_update_time': time.time()})
                self.stats['curr_sample'] = this_gulp_time
                self.update_stats()
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                if self.command_vals['dest_ip'] != '0.0.0.0':
                    start_time = time.time()
                    """
                    idata = ispan.data.view('cf32').reshape([nchan, nbeamset, nbeampol, self.ntime_gulp])
                    # Tranpose to dual-pol beam x time x chan x pol
                    try:
                        Transpose(tdata, idata, axes=(1,3,0,2))
                    except NameError:
                        tdata = BFArray(shape=(nbeamset,self.ntime_gulp,nchan,nbeampol), dtype='cf32', space='cuda')
                        Transpose(tdata, idata, axes=(1,3,0,2))
                    # Downselect beams and copy to CPU
                    ddata = tdata[0,...]
                    try:
                        copy_array(idata_cpu, ddata)
                    except NameError:
                        idata_cpu = ddata.copy(space='cuda_host')
                    idata_cpu_r = idata_cpu.reshape(self.ntime_gulp, 1, nchan*nbeampol)
                    """
                    idata = ispan.data.view('cf32').reshape([nchan, nbeam, self.ntime_gulp])
                    # Downselect beams and copy to CPU
                    # Transpose to time x chan x beam order
                    idata_cpu = idata[:,0:(self.npol // npol) * self.nbeam_send,:].copy(space='system').transpose([2,0,1]).copy(space='system')
                    idata_cpu_r = idata_cpu.reshape(self.ntime_gulp, 1, nchan*self.nbeam_send*(self.npol // npol))
                    burst_bits = self._npacket_burst * nchan * self.nbeam_send * self.npol // npol * 2 * 32 
                    try:
                        toff = 0
                        while(toff < self.ntime_gulp):
                            t0 = time.time()
                            udt.send(desc, this_gulp_time + toff, 1, self.pipeline_idx-1, 0, idata_cpu_r[toff : toff + self._npacket_burst])
                            toff += self._npacket_burst
                            dt = time.time() - t0
                            delay = (burst_bits / (self._max_bps)) - dt
                            if delay > 0:
                                time.sleep(delay)

                    except Exception as e:
                        self.log.error("VLBI OUTPUT >> Sending error: %s" % str(e))
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
                this_gulp_time += self.ntime_gulp
