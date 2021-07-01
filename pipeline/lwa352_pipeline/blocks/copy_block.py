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
import ujson as json
import numpy as np

from .block_base import Block

class Copy(Block):
    """
    **Functionality**
    
    This block copies data from one bifrost buffer to another.
    The buffers may be in CPU or GPU space.
    
    **New Sequence Condition**
    
    This block has no new sequence condition. It will output a new sequence
    only when the upstream sequence changes.
    
    **Input Header Requirements**
    
    This block has no input header requirements.
    
    **Output Headers**
    
    This block copies input headers downstream, adding none of its own.

    **Data Buffers**

    *Input Data Buffer*: A bifrost ring buffer of at least
    ``nbyte_per_time x ntime_gulp`` bytes size.

    *Output Data Buffer*: A bifrost ring buffer whose size is set by the
    ``buffer_multiplier`` and/or ``buf_size_gbytes`` parameters.

    **Instantiation**

    :param log: Logging object to which runtime messages should be
        emitted.
    :type log: logging.Logger

    :param iring: bifrost input data ring
    :type iring: bifrost.ring.Ring

    :param oring: bifrost output data ring
    :type oring: bifrost.ring.Ring

    :param guarantee: If True, read data from the input ring in blocking "guaranteed"
        mode, applying backpressure if necessary to ensure no data are missed by this block.
    :type guarantee: Bool

    :param core: CPU core to which this block should be bound. A value of -1 indicates no binding.
    :type core: int

    :param gpu: GPU device ID to which this block should copy to/from.
        A value of -1 indicates no binding. This parameter need not be provided if neither
        input nor output data rings are allocated in GPU (or GPU-pinned) memory.
    :type gpu: int

    :param ntime_gulp: Number of time samples to copy with each gulp.
    :type ntime_gulp: int

    :param nbyte_per_time: Number of bytes per time sample. The total number of bytes read
        with each gulp is ``nbyte_per_time x ntime_gulp``.
    :type nbyte_per_time: int

    :param buffer_multiplier: The block will set the output buffer size to be
        ``4 x buffer_multiplier`` times the size of a single data gulp. If
        ``buf_size_gbytes`` is also provided then the output memory block size is required
        to be a mulitple of ``buffer_multiplier x ntime_gulp x nbyte_per_time`` bytes.
    :type buffer_multiplier: int

    :param buf_size_gbytes: If provided, attempt to set the output buffer size to
        ``buf_size_gbytes`` Gigabytes (10**9 Bytes). Round down the buffer such that the
        chosen size is the largest integer multiple of ``buffer_multiplier x ntime_gulp x nbyte_per_time``
        which is less than ``10**9 x buf_size_gbytes``.
        Note that bifrost (seems to) round up buffer sizes to an integer power of two
        number of bytes, so be careful about accidentally allocating more memory than you have!
    :type buf_size_gbytes: int

    """

    def __init__(self, log, iring, oring, ntime_gulp=2500, buffer_multiplier=1,
                 guarantee=True, core=-1, nbyte_per_time=184*352*2, gpu=-1,
                 buf_size_gbytes=None):

        super(Copy, self).__init__(log, iring, oring, guarantee, core, etcd_client=None)
        cpu_affinity.set_core(self.core)
        self.ntime_gulp = ntime_gulp
        self.gpu = gpu

        if self.gpu != -1:
            BFSetGPU(self.gpu)

        self.buffer_multiplier = buffer_multiplier
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nbyte_per_time*1        # complex8
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
                                #idata = ispan.data_view('ci4').reshape(self.ntime_gulp, ihdr['nchan'], ihdr['nstand'], ihdr['npol'])
                                #print(idata[0:10,0:10,0,0])
                                stream_synchronize()

                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': 8*self.igulp_size / process_time / 1e9})
