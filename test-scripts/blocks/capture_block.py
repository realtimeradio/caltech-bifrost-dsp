from bifrost.proclog import ProcLog
import bifrost.affinity as cpu_affinity
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture

import time
import simplejson as json
import threading
import ctypes
import numpy as np

class Capture(object):
    time_tag = 0
    def __init__(self, log, *args, **kwargs):
        self.log    = log
        self.args   = args
        self.kwargs = kwargs
        self.utc_start = self.kwargs['utc_start']
        del self.kwargs['utc_start']
        self.shutdown_event = threading.Event()
        ## HACK TESTING
        #self.seq_callback = None
    def shutdown(self):
        self.shutdown_event.set()
    def seq_callback(self, seq0, chan0, nchan, nsrc,
                     time_tag_ptr, hdr_ptr, hdr_size_ptr):
        timestamp0 = 0
        time_tag0  = 0
        self.time_tag += 1
        time_tag   = self.time_tag
        #print("++++++++++++++++ seq0     =", seq0)
        #print("                 time_tag =", time_tag)
        time_tag_ptr[0] = time_tag
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'nchan':    nchan,
               'cfreq':    (chan0 + 0.5*(nchan-1))*1,
               'bw':       nchan*1,
               'nstand':   nsrc*16,
               #'stand0':   src0*16, # TODO: Pass src0 to the callback too(?)
               'npol':     2,
               'complex':  True,
               'nbit':     4}
        hdr_str = json.dumps(hdr).encode()
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        self.header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(self.header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_snap2(self.seq_callback)
        with UDPCapture(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
                #print status
        del capture