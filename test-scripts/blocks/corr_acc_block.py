import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap

import time
import json
import numpy as np

class CorrAcc(object):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring, oring, ntime_gulp=2400,
                 guarantee=True, core=-1, nchans=192, npols=704, acc_len=24000):
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        self.log = log
        self.iring = iring
        self.oring = oring
        self.guarantee = guarantee
        self.core = core
        self.ntime_gulp = ntime_gulp
        self.acc_len = acc_len
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = 47849472 * 8 # complex64
        self.ogulp_size = self.igulp_size
        # integration buffer
        self.accdata = BFArray(shape=(self.igulp_size // 4), dtype='i32', space='cuda')
        self.bfbf = LinAlg()

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        oseq = None
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                subacc_id = ihdr['seq0'] % self.acc_len
                first = subacc_id == 0
                last = subacc_id == self.acc_len - self.ntime_gulp
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                for ispan in iseq.read(self.igulp_size):
                    self.log.debug("Accumulating correlation")
                    idata = ispan.data_view('i32')
                    if first:
                        oseq = oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                        # TODO: surely there are more sensible ways to implement a vacc
                        BFMap("a = b", data={'a': self.accdata, 'b': idata})
                    else:
                        BFMap("a += b", data={'a': self.accdata, 'b': idata})
                    if last:
                        if oseq is None:
                            print("Skipping output because oseq isn't open")
                        else:
                            # copy to CPU
                            odata = ospan.data_view('i32')
                            odata = self.accdata
                            print(odata[0:10])
                            ospan.close()
                            oseq.end()
            # If upstream process stops producing, close things gracefully
            # TODO: why is this necessary? Get exceptions from ospan.__exit__ if not here
            if oseq:
                ospan.close()
                oseq.end()
