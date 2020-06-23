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

class Copy(object):
    """
    Copy data from one buffer to another.
    """
    def __init__(self, log, iring, oring, ntime_gulp=2500,
                 guarantee=True, core=-1, nchans=192, npols=704, gpu=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.guarantee = guarantee
        self.core = core
        self.gpu = gpu

        if self.gpu != -1:
            BFSetGPU(self.gpu)

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

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.igulp_size)
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet) as oseq:
                    for ispan in iseq.read(self.igulp_size):
                        with oseq.reserve(self.igulp_size) as ospan:
                            #self.log.debug("Copying output")
                            #odata = ospan.data_view('ci4')
                            copy_array(ospan.data, ispan.data)
                            # The copy is asynchronous, so we must wait for it to finish
                            # before committing this span
                            stream_synchronize()
