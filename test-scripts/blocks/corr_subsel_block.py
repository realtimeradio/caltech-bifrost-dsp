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

class CorrSubSel(object):
    """
    Grab arbitrary entries from a GPU buffer and copy them to the CPU
    """
    def __init__(self, log, iring, oring,
            guarantee=True, core=-1, nchans=192, gpu=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.guarantee = guarantee
        self.core = core
        self.nchans = nchans
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
        self.igulp_size = 47849472 * 8 # complex64

        self.subsel = BFArray(np.array(list(range(4656)), dtype=np.int32), dtype='i32', space='cuda')
        self.nvis_out = len(self.subsel)
        self.obuf_gpu = BFArray(shape=[self.nchans, self.nvis_out], dtype='i64', space='cuda')
        self.ogulp_size = self.nchans * self.nvis_out * 8

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        oseq = None
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                for ispan in iseq.read(self.igulp_size):
                    self.log.debug("Grabbing subselection")
                    idata = ispan.data_view('i64').reshape(47849472)
                    with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet) as oseq:
                        with oseq.reserve(self.ogulp_size) as ospan:
                            rv = _bf.bfXgpuSubSelect(idata.as_BFarray(), self.obuf_gpu.as_BFarray(), self.subsel.as_BFarray())
                            if (rv != _bf.BF_STATUS_SUCCESS):
                                self.log.error("xgpuIntialize returned %d" % rv)
                                raise RuntimeError
                            odata = ospan.data_view(dtype='i64').reshape([self.nchans, self.nvis_out])
                            copy_array(odata, self.obuf_gpu)
                            # Wait for copy to complete before committing span
                            stream_synchronize()
