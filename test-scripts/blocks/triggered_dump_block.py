import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan

import time
import simplejson as json
import numpy as np

from blocks.block_base import Block

class TriggeredDump(Block):
    """
    Dump a buffer to disk
    """
    def __init__(self, log, iring, ntime_gulp=2500,
                 guarantee=True, core=-1, nchans=192, npols=704, etcd_client=None):

        super(TriggeredDump, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        self.ntime_gulp = ntime_gulp

        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nchans*npols*1        # complex8

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            ohdr = ihdr.copy()
            # Mash header in here if you want
            ohdr_str = json.dumps(ohdr)
            prev_time = time.time()
            for ispan in iseq.read(self.igulp_size):
                if ispan.size < self.igulp_size:
                    continue # Ignore final gulp
                #curr_time = time.time()
                #acquire_time = curr_time - prev_time
                #prev_time = curr_time
                #curr_time = time.time()
                #reserve_time = curr_time - prev_time
                #prev_time = curr_time
                ## do stuff
                #        
                #curr_time = time.time()
                #process_time = curr_time - prev_time
                #prev_time = curr_time
                #self.perf_proclog.update({'acquire_time': acquire_time, 
                #                          'reserve_time': reserve_time, 
                #                          'process_time': process_time,
                #                          'gbps': 8*self.igulp_size / process_time / 1e9})
