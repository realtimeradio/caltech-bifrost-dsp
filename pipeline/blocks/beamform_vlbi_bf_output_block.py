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
import simplejson as json
import socket
import struct
import numpy as np

from blocks.block_base import Block

class BeamformVlbiOutput(Block):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring,
                 guarantee=True, core=-1, etcd_client=None, dest_port=10000,
                 ntime_gulp=480,
                 ):
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        super(BeamformVlbiOutput, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        cpu_affinity.set_core(self.core)

        #self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = Address('100.100.100.1', 7777)
        self.sock = UDPSocket()
        self.sock.connect(addr)

        self.dest_ip = '0.0.0.0'
        self.new_dest_ip = '0.0.0.0'
        self.dest_port = dest_port
        self.new_dest_port = dest_port
        self.packet_delay_ns = 1
        self.new_packet_delay_ns = 1
        self.update_pending = True
        self.ntime_gulp = ntime_gulp

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
        if 'packet_delay_ns' in v:
            self.new_packet_delay_ns = v['packet_delay_ns']
        self.update_pending = True
        self.stats.update({'new_dest_ip': self.new_dest_ip,
                           'new_dest_port': self.new_dest_port,
                           'new_packet_delay_ns': self.new_packet_delay_ns,
                           'update_pending': self.update_pending,
                           'last_cmd_time': time.time()})
        self.update_stats()

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        prev_time = time.time()

        nbeam_max = 1
        nchan_max = 184
        dout = BFArray(shape=[self.ntime_gulp, 2, 184], dtype='cf32', space='cuda_host')
        #with UDPTransmit('lwa352_vbeam_%d' % (nchan_max), sock=self.sock, core=self.core) as udt:
        with UDPTransmit('lwa352_vbeam_%d' % (nchan_max), sock=self.sock, core=self.core) as udt:
            desc = HeaderInfo()
            desc.set_nsrc(1)
            for iseq in self.iring.read():
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("Retransmit: Start of new sequence: %s", str(ihdr))
                
                chan0   = ihdr['chan0']
                nchan   = ihdr['nchan']
                nbeam   = ihdr['nbeam']
                npol    = ihdr['npol']
                igulp_size = self.ntime_gulp*nchan*nbeam*8        # complex64
                igulp_shape = (self.ntime_gulp,nchan,nbeam)
                
                seq0 = ihdr['seq0']
                seq = seq0
                
                desc.set_nchan(nchan)
                desc.set_chan0(chan0)
                
                prev_time = time.time()
                for ispan in iseq.read(igulp_size):
                    if ispan.size < igulp_size:
                        continue # Ignore final gulp
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    
                    dout[...] = ispan.data_view('cf32').reshape(self.ntime_gulp, nbeam//2, 2, nchan)[:,0,:,:]
                    dout_reshape = dout.reshape(self.ntime_gulp, 1, 2*nchan)
                    try:
                        pass
                        udt.send(desc, seq, 1, 0, 0, dout_reshape)
                    except Exception as e:
                        print(type(self).__name__, 'Sending Error', str(e))
                        
                    seq += self.ntime_gulp
                    
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': -1, 
                                              'process_time': process_time,})
                    
        del udt
