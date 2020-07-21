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

class CorrAcc(Block):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring, oring, ntime_gulp=2400,
                 guarantee=True, core=-1, nchans=192, npols=704, acc_len=24000, gpu=-1, etcd_client=None):
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        super(CorrAcc, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)
        self.ntime_gulp = ntime_gulp
        self.acc_len = acc_len
        self.gpu = gpu

        if self.gpu != -1:
            BFSetGPU(self.gpu)
        
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = 47849472 * 8 # complex64
        self.ogulp_size = self.igulp_size
        # integration buffer
        self.accdata = BFArray(shape=(self.igulp_size // 4), dtype='i32', space='cuda')
        self.bfbf = LinAlg()

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        oseq = None
        process_time = 0
        with self.oring.begin_writing() as oring:
            prev_time = time.time()
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                subacc_id = ihdr['seq0'] % self.acc_len
                first = subacc_id == 0
                last = subacc_id == self.acc_len - self.ntime_gulp
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                for ispan in iseq.read(self.igulp_size):
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    self.log.debug("Accumulating correlation")
                    idata = ispan.data_view('i32')
                    if first:
                        oseq = oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        # TODO: surely there are more sensible ways to implement a vacc
                        BFMap("a = b", data={'a': self.accdata, 'b': idata})
                    else:
                        BFMap("a += b", data={'a': self.accdata, 'b': idata})
                    curr_time = time.time()
                    process_time += curr_time - prev_time
                    prev_time = curr_time
                    if last:
                        if oseq is None:
                            print("Skipping output because oseq isn't open")
                        else:
                            # copy to CPU
                            odata = ospan.data_view('i32').reshape(self.accdata.shape)
                            copy_array(odata, self.accdata)
                            # Wait for copy to complete before committing span
                            stream_synchronize()
                            ospan.close()
                            oseq.end()
                            curr_time = time.time()
                            process_time += curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})
                            process_time = 0
            # If upstream process stops producing, close things gracefully
            # TODO: why is this necessary? Get exceptions from ospan.__exit__ if not here
            if oseq:
                ospan.close()
                oseq.end()
