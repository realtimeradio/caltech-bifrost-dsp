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

from blocks.block_base import Block

class CorrOutputFull(Block):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring,
                 guarantee=True, core=-1, nchans=192, npols=2, nstands=352, etcd_client=None, dest_port=10000):
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        super(CorrOutputFull, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        self.nchans = nchans
        self.npols = npols
        self.nstands = nstands
        self.matlen = nchans * (nstands//2+1)*(nstands//4)*npols*npols*4

        self.igulp_size = self.matlen * 8 # complex64

        # Arrays to hold the conjugation and bl indices of data coming from xGPU
        self.antpol_to_bl = BFArray(np.zeros([nstands, nstands, npols, npols], dtype=np.int32), space='system')
        self.bl_is_conj   = BFArray(np.zeros([nstands, nstands, npols, npols], dtype=np.int32), space='system')
        self.reordered_data = BFArray(np.zeros([nstands, nstands, npols, npols, nchans, 2], dtype=np.int32), space='system')

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.dest_ip = None
        self.new_dest_ip = None
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
        self.stats_proclog.update({'new_dest_ip': self.new_dest_ip,
                                   'new_dest_port': self.new_dest_port,
                                   'update_pending': self.update_pending,
                                   'last_cmd_time': time.time()})

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        prev_time = time.time()
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            upstream_acc_len = ihdr['acc_len']
            upstream_start_time = ihdr['start_time']
            self.antpol_to_bl[...] = ihdr['ant_to_bl_id']
            self.bl_is_conj[...] = ihdr['bl_is_conj']
            for ispan in iseq.read(self.igulp_size):
                print('CORR OUTPUT >> reordering')
                # Update destinations if necessary
                if self.update_pending:
                    self.dest_ip = self.new_dest_ip
                    self.dest_port = self.new_dest_port
                    self.update_pending = False
                    self.log.info("CORR OUTPUT >> Updating destination to %s:%s" % (self.dest_ip, self.dest_port))
                    self.stats_proclog.update({'dest_ip': self.dest_ip,
                                               'dest_port': self.dest_port,
                                               'update_pending': self.update_pending,
                                               'last_update_time': time.time()})
                self.stats_proclog.update({'curr_sample': this_gulp_time})
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                _bf.bfXgpuReorder(ispan.data.as_BFarray(), self.reordered_data.as_BFarray(), self.antpol_to_bl.as_BFarray(), self.bl_is_conj.as_BFarray())
                if self.dest_ip is not None:
                    dout = np.zeros([self.npols, self.npols, self.nchans, 2], dtype='>i')
                    for stand0 in range(self.nstands):
                        for stand1 in range(stand0, self.nstands):
                            header = struct.pack(">Q3L", this_gulp_time, ihdr['chan0'], stand0, stand1)
                            dout = self.reordered_data[stand0, stand1]
                            self.sock.sendto(header + dout.tobytes(), (self.dest_ip, self.dest_port))
                    self.log.info("CORR OUPUT >> Sending complete")
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.stats_proclog.update({'last_end_sample': this_gulp_time})
                # And, update overall time counter
                this_gulp_time += upstream_acc_len
