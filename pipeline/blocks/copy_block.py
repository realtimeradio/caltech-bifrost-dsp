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

class Copy(Block):
    """
    Copy data from one buffer to another.
    """
    def __init__(self, log, iring, oring, ntime_gulp=2500, buffer_multiplier=1,
                 guarantee=True, core=-1, nchan=192, nstand=352, npol=2, gpu=-1, etcd_client=None,
                 buf_size_gbytes=None):

        super(Copy, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)
        cpu_affinity.set_core(self.core)
        self.ntime_gulp = ntime_gulp
        self.gpu = gpu

        if self.gpu != -1:
            BFSetGPU(self.gpu)

        self.buffer_multiplier = buffer_multiplier
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nchan*nstand*npol*1        # complex8
        # round down buffer size to an integer gulps
        if buf_size_gbytes is None:
            self.buf_size = 4 * self.igulp_size*self.buffer_multiplier
        else:
            self.buf_size = int(1e9 * buf_size_gbytes) // (self.igulp_size * self.buffer_multiplier) * self.igulp_size * self.buffer_multiplier

        # Looking at bifrost's ring_impl.cpp the buffer will get rounded up to a power of 2
        buf_size_rounded = 2**int(np.ceil(np.log2(self.buf_size)))
        self.log.info("COPY >> Allocating %.2f GB of memory (which will be rounded up by bifrost to %.2f GB" % (self.buf_size / 1e9, buf_size_rounded / 1e9))
        self.oring.resize(self.buffer_multiplier*self.igulp_size, total_span=self.buf_size)
        self.log.info("COPY >> Allocation complete")

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})


        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet) as oseq:
                    for ispan in iseq.read(self.igulp_size):
                        if ispan.size < self.igulp_size:
                            continue # Ignore final gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time
                        with oseq.reserve(self.igulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time
                            # The copy to a GPU is asynchronous, so we must wait for it to finish
                            # before committing this span
                            copy_array(ospan.data, ispan.data)
                            if (self.oring.space == 'cuda') or (self.iring.space=='cuda'):
                                stream_synchronize()

                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': 8*self.igulp_size / process_time / 1e9})
