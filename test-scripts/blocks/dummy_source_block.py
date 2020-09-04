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
                 core=-1, nchans=192, nstands=352, npols=2, skip_write=False, target_throughput=22.0, testfile=None):
        self.log = log
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.core = core
        self.nchans = nchans
        self.npols = npols
        self.nstands = 352
        self.ninputs = nstands * npols
        self.skip_write = skip_write
        self.target_throughput = target_throughput
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.gulp_size = self.ntime_gulp*nchans*nstands*npols*1        # complex8

        # file containing test data
        if testfile is not None:
            self.testfile = open(testfile, 'rb')
        else:
            self.testfile = None

        # make an array ninputs-elements long with [station, pol] IDs.
        # e.g. if input_to_ant[12] = [27, 1], then the 13th input is stand 27, pol 1
        self.input_to_ant = np.zeros([self.ninputs, 2], dtype=np.int32)
        for s in range(self.nstands):
            for p in range(self.npols):
                self.input_to_ant[self.npols*s + p] = [s, p]

        self.ant_to_input = np.zeros([self.nstands, self.npols], dtype=np.int32)
        for i, inp in enumerate(self.input_to_ant):
            stand = inp[0]
            pol = inp[1]
            self.ant_to_input[stand, pol] = i

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
        hdr['chan0'] = 0
        hdr['nstand'] = self.nstands
        hdr['npol'] = self.npols
        hdr['seq0'] = 0
        hdr['input_to_ant'] = self.input_to_ant.tolist()
        hdr['ant_to_input'] = self.ant_to_input.tolist()
        time_tag = 0
        REPORT_PERIOD = 100
        bytes_per_report = REPORT_PERIOD * self.gulp_size
        acquire_time = 0 # this block doesn't have an input ring
        gbps = 0
        extra_delay = 0
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
                            if self.testfile:
                                # read data from file, and at the end of the file, cycle back to the beginning
                                rawdata = self.testfile.read(self.gulp_size)
                                if len(rawdata) != self.gulp_size:
                                    self.testfile.seek(0)
                                    rawdata = self.testfile.read(self.gulp_size)
                                if len(rawdata) != self.gulp_size:
                                    self.log.error("Failed to read input data file")
                                else:
                                    self.test_data[time_tag % NTEST_BLOCKS] = np.frombuffer(rawdata, dtype=np.uint8).reshape(self.test_data.shape[1:])

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
                    time.sleep(max(0, extra_delay / REPORT_PERIOD))
                    if time_tag % REPORT_PERIOD == 0:
                        tock = time.time()
                        dt = tock - tick
                        gbps = 8*bytes_per_report / dt / 1e9
                        self.log.info('%d: Sent %d bytes in %.2f seconds (%.2f Gb/s)' % (time_tag // REPORT_PERIOD, bytes_per_report, dt, gbps))
                        target_time = 8*bytes_per_report / self.target_throughput / 1e9
                        extra_delay = target_time - dt + extra_delay
                        tick = tock
        if self.testfile:
            self.testfile.close()
