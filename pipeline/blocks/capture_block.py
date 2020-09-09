from bifrost.proclog import ProcLog
import bifrost.affinity as cpu_affinity
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture, UDPVerbsCapture

import time
import simplejson as json
import threading
import ctypes
import numpy as np

class Capture(object):
    time_tag = 0
    def __init__(self, log, fs_hz=196000000, chan_bw_hz=23925.78125,
                     input_to_ant=None, nstands=352, npols=2,
                     *args, **kwargs):
        self.log    = log
        self.fs_hz  = fs_hz # sampling frequency in Hz
        self.chan_bw_hz = chan_bw_hz # Channel bandwidth in Hz
        self.args   = args
        self.kwargs = kwargs
        self.core = self.kwargs.get('core', 0)
        self.utc_start = self.kwargs['utc_start']
        self.nstands = nstands
        self.npols = npols
        if 'ibverbs' in self.kwargs:
            if self.kwargs['ibverbs']:
                self.CaptureClass = UDPVerbsCapture
            else:
                self.CaptureClass = UDPCapture
            del self.kwargs['ibverbs']
        else:
            self.CaptureClass = UDPCapture

        del self.kwargs['utc_start']
        # Add gulp size = slot_ntime requirement which is special
        # for the LWA352 receiver
        self.kwargs['slot_ntime'] = kwargs['buffer_ntime']
        self.shutdown_event = threading.Event()

        # make an array ninputs-elements long with [station, pol] IDs.
        # e.g. if input_to_ant[12] = [27, 1], then the 13th input is stand 27, pol 1
        if input_to_ant is not None:
            self.input_to_ant = input_to_ant
        else:
            self.input_to_ant = np.zeros([nstands*npols, 2], dtype=np.int32)
            for s in range(nstands):
                for p in range(npols):
                    self.input_to_ant[npols*s + p] = [s, p]

        self.ant_to_input = np.zeros([nstands, npols], dtype=np.int32)
        for i, inp in enumerate(self.input_to_ant):
            stand = inp[0]
            pol = inp[1]
            self.ant_to_input[stand, pol] = i
        self.time_tag = 0
           
        ## HACK TESTING
        #self.seq_callback = None
    def shutdown(self):
        self.shutdown_event.set()
    def seq_callback(self, seq0, chan0, nchan, nsrc,
                     time_tag_ptr, hdr_ptr, hdr_size_ptr):
        #t0 = time.time()
        self.time_tag += 1
        sync_time = time_tag_ptr[0]
        #print("++++++++++++++++ seq0     =", seq0)
        #print("                 time_tag =", self.time_tag)
        #print("                 sync_time =", time.ctime(sync_time))
        #self.log.info("Capture >> New sequence at %s" % time.ctime())
        time_tag_ptr[0] = self.time_tag
        nchan = nchan * (nsrc * 32 // self.nstands)
        hdr = {'time_tag': self.time_tag,
               'sync_time': sync_time,
               'seq0':     seq0, 
               'chan0':    chan0,
               'nchan':    nchan,
               'fs_hz':    self.fs_hz,
               'sfreq':    chan0*self.chan_bw_hz,
               'bw_hz':    nchan*self.chan_bw_hz,
               'nstand':   self.nstands,
               'input_to_ant': self.input_to_ant.tolist(),
               'ant_to_input': self.ant_to_input.tolist(),
               #'stand0':   src0*16, # TODO: Pass src0 to the callback too(?)
               'npol':     self.npols,
               'complex':  True,
               'nbit':     4}
        #if self.input_to_ant.shape != (nstand, npol):
        #    self.log.error("Input order shape %s does not match data stream (%d, %d)" %
        #                    (self.input_to_ant.shape, nstand, npol))

        hdr_str = json.dumps(hdr).encode()
        #hdr_str = b'\x00\x00\x00\x00'
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        self.header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(self.header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        #t1 = time.time()
        #print(t1-t0)
        return 0
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_snap2(self.seq_callback)
        with self.CaptureClass(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
                #print status
        del capture
