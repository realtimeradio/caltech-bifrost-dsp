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
import socket

from blocks.block_base import Block

class Vacc(Block):
    """
    Copy data from one buffer to another.
    """
    def __init__(self, log, iring, oring, ninput_beam=16, beam_id=0,
                 guarantee=True, core=-1, nchans=192, nint=16, etcd_client=None):

        super(Vacc, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)
        self.ntime_gulp = nint
        self.ninput_beam = ninput_beam
        self.nchans = nchans
        self.beam_id = beam_id

        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*ninput_beam*nchans*4*4  # XX, YY, real(XY), im(XY) x 32-bit float
        self.ogulp_size = self.ninput_beam*nchans*4*4  # XX, YY, real(XY), im(XY) x 32-bit float
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.vacc = BFArray(np.zeros([self.ninput_beam, nchans, 4]), dtype='f32', space='system')

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        with self.oring.begin_writing() as oring:
            #with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet) as oseq:
            with oring.begin_sequence(time_tag=0, header='', nringlet=1) as oseq:
                for iseq in self.iring.read(guarantee=self.guarantee):
                    ihdr = json.loads(iseq.header.tostring())
                    ohdr = ihdr.copy()
                    # Mash header in here if you want
                    ohdr_str = json.dumps(ohdr)
                    prev_time = time.time()
                    for ispan in iseq.read(self.igulp_size):
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time
                        with oseq.reserve(self.ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time
                            #self.log.debug("Copying output")
                            odata = ospan.data_view(np.float32)
                            idata = ispan.data_view(np.float32).reshape([self.ninput_beam, self.ntime_gulp, self.nchans, 4])
                            for i in range(self.ninput_beam):
                                 self.vacc[i] += idata[i].sum(axis=0)
                                 #    self.s.sendto(idata[i,:,:,:].sum(axis=0).tobytes(), ('127.0.0.1', 10001))
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': 8*self.igulp_size / process_time / 1e9})
