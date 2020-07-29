from bifrost.proclog import ProcLog
import bifrost.ndarray as BFArray
import bifrost.affinity as cpu_affinity

import time
import simplejson as json
import threading
import numpy as np

NTEST_BLOCKS = 2

class DummySource(object):
    """
    A dummy source block for throughput testing. Does nothing
    but mark input buffers ready for consumption.
    """
    def __init__(self, log, oring, ntime_gulp=2500,
                 core=-1, nchans=192, nstands=352, npols=2, skip_write=False):
        self.log = log
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.core = core
        self.nchans = nchans
        self.npols = npols
        self.nstands = 352
        self.skip_write = skip_write
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.gulp_size = self.ntime_gulp*nchans*nstands*npols*1        # complex8

        if skip_write:
            self.test_data = BFArray(shape=[NTEST_BLOCKS, ntime_gulp, nchans, nstands, npols], dtype='i8', space='system')
        else:
            print("initializing random numbers")
            #TODO Can't get 'ci4' type to behave
            self.test_data = BFArray(np.random.randint(0, high=255, size=[NTEST_BLOCKS, ntime_gulp, nchans, nstands, npols]),
                                dtype='u8', space='system')
            #self.test_data = BFArray(np.ones([NTEST_BLOCKS, ntime_gulp, nchans, nstands, npols]),
            #                    dtype='u8', space='system')
            #self.test_data[:,:,:,:,0] = 0

        self.shutdown_event = threading.Event()

    def get_test_data(self):
        r = self.test_data >> 4
        i = self.test_data & 0xf
        return r + 1j*i

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        time.sleep(0.1)
        self.oring.resize(self.gulp_size, self.gulp_size*4)
        hdr = {}
        hdr['nchan'] = self.nchans
        hdr['nstand'] = self.nstands
        hdr['npol'] = self.npols
        hdr['seq0'] = 0
        time_tag = 0
        REPORT_PERIOD = 100
        bytes_per_report = REPORT_PERIOD * self.gulp_size
        acquire_time = 0 # this block doesn't have an input ring
        gbps = 0
        with self.oring.begin_writing() as oring:
            tick = time.time()
            ohdr_str = json.dumps(hdr)
            prev_time = time.time()
            with oring.begin_sequence(time_tag=time_tag, header=ohdr_str) as oseq:
    	        while not self.shutdown_event.is_set():
                    with oseq.reserve(self.gulp_size) as ospan:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        if not self.skip_write:
                            odata = ospan.data_view(shape=self.test_data.shape[1:], dtype=self.test_data.dtype)
                            odata[...] = self.test_data[time_tag % NTEST_BLOCKS]
                        time_tag += 1
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': reserve_time, 
                                              'process_time': process_time,
                                              'gbps' : gbps})
                    if time_tag % REPORT_PERIOD == 0:
                        tock = time.time()
                        dt = tock - tick
                        gbps = 8*bytes_per_report / dt / 1e9
                        print('Send %d bytes in %.2f seconds (%.2f Gb/s)' % (bytes_per_report, dt, gbps))
                        tick = tock
