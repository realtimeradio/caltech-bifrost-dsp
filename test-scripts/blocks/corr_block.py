import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan

import time
import json
import numpy as np

class Corr(object):
    """
    Perform cross-correlation using xGPU
    """
    def __init__(self, log, iring, oring, ntime_gulp=2500,
                 guarantee=True, core=-1, nchans=192, npols=704, acc_len=2400):
        assert (acc_len % ntime_gulp == 0), "Acculmulation length must be a multiple of gulp size"
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.guarantee = guarantee
        self.core = core
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
        self.igulp_size = self.ntime_gulp*nchans*npols*1        # complex8
        self.ogulp_size = 47849472 * 8 # complex64

        # initialize xGPU. Arrays passed as inputs don't really matter here
        # but we need to pass something
        ibuf = BFArray(0, dtype='i8', space='cuda')
        obuf = BFArray(0, dtype='i64', space='cuda')
        rv = _bf.xgpuInitialize(ibuf.as_BFarray(), obuf.as_BFarray(), 0)
        if (rv != _bf.BF_STATUS_SUCCESS):
            self.log.error("xgpuIntialize returned %d" % rv)
            raise RuntimeError

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        oseq = None
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                self.log.debug("Correlating output")
                ihdr = json.loads(iseq.header.tostring())
                subacc_id = ihdr['seq0'] % self.acc_len
                first = subacc_id == 0
                last = subacc_id == self.acc_len - self.ntime_gulp
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                for ispan in iseq.read(self.igulp_size):
                    if first:
                        oseq = oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                    _bf.xgpuKernel(ispan.data.as_BFarray(), ospan.data.as_BFarray(), int(last))
                    if last:
                        if oseq is None:
                            print("CORR >> Skipping output because oseq isn't open")
                        else:
                            ospan.close()
                            oseq.end()
            # If upstream process stops producing, close things gracefully
            # TODO: why is this necessary? Get exceptions from ospan.__exit__ if not here
            if oseq:
                ospan.close()
                oseq.end()

