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
import numpy as np

from blocks.block_base import Block

class CorrOutput(Block):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring,
                 guarantee=True, core=-1, nchans=192, npols=2, nstands=352, etcd_client=None):
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        super(CorrOutput, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        self.nchans = nchans
        self.npols = npols
        self.nstands = nstands
        self.matlen = nchans * (nstands//2+1)*(nstands//4)*npols*npols*4

        self.igulp_size = self.matlen * 8 # complex64

        ## Arrays to hold the conjugation and bl indices of data coming from xGPU
        #self.antpol_to_bl = BFArray(np.zeros([nstands, nstands, npols, npols], dtype=np.int32), space='system')
        #self.bl_is_conj   = BFArray(np.zeros([nstands, nstands, npols, npols], dtype=np.int32), space='system')
        #self.reordered_data = BFArray(np.zeros([nchans, nstands, nstands, npols, npols, 2], dtype=np.int32), space='system')


    def main(self):
        while(True):
            continue
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
                self.stats_proclog.update({'curr_sample': this_gulp_time})
                curr_time = time.time()
                reserve_time = curr_time - prev_time
                prev_time = curr_time
                _bf.bfXgpuReorder(self.accdata_h.as_BFarray(), self.reordered_data.as_BFarray(), self.antpol_to_bl.as_BFarray(), self.bl_is_conj.as_BFarray())
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.stats_proclog.update({'last_end_sample': this_gulp_time})
                # And, update overall time counter
                this_gulp_time += upstream_acc_len
