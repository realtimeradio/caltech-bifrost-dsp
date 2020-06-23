from bifrost.proclog import ProcLog
import bifrost.ndarray as BFArray
import bifrost.affinity as cpu_affinity

import time
import simplejson as json
import threading
import numpy as np

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

        self.test_data = BFArray(1*np.ones(self.gulp_size), dtype='u8', space='system')
        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        time.sleep(0.1)
        self.oring.resize(self.gulp_size)
        hdr = {}
        hdr['nchan'] = self.nchans
        hdr['nstand'] = self.nstands
        hdr['npol'] = self.npols
        hdr['seq0'] = 0
        time_tag = 0
        REPORT_PERIOD = 100
        bytes_per_report = REPORT_PERIOD * self.gulp_size
        with self.oring.begin_writing() as oring:
            tick = time.time()
            while not self.shutdown_event.is_set():
                ohdr_str = json.dumps(hdr)
                with oring.begin_sequence(time_tag=time_tag, header=ohdr_str) as oseq:
                    with oseq.reserve(self.gulp_size) as ospan:
                        if not self.skip_write:
                            ospan.data[...] = self.test_data
                        time_tag += 1
                        hdr['seq0'] += self.ntime_gulp
                if time_tag % REPORT_PERIOD == 0:
                    tock = time.time()
                    dt = tock - tick
                    print('Send %d bytes in %.2f seconds (%.2f Gb/s)' % (bytes_per_report, dt, (8*bytes_per_report / dt / 1e9)))
                    tick = tock
