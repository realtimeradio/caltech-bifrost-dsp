from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture
from bifrost.packet_writer import HeaderInfo, DiskWriter, UDPTransmit
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.unpack import unpack as Unpack
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.memory import memcpy as BFMemCopy, memset as BFMemSet
from bifrost.linalg import LinAlg
from bifrost import map as BFMap, asarray as BFAsArray
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, stream_synchronize as BFSync, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()

#import numpy as np
import signal
import logging
import time
import os
import argparse
import ctypes
import threading
import json
import socket
import struct
#import time
import datetime
from collections import deque

ACTIVE_COR_CONFIG = threading.Event()

__version__    = "0.2"
__date__       = '$LastChangedDate: 2016-08-09 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2016, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

class CaptureOp(object):
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
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
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

class DummyOp(object):
    def __init__(self, log, iring, guarantee=True, core=-1, ntime_gulp=128):
        self.log   = log
        self.iring = iring
        self.guarantee = guarantee
        self.core = core
        self.ntime_gulp = ntime_gulp
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        
    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        igulp_size = self.ntime_gulp*64*704*1        # complex8
        for iseq in self.iring.read():
            self.log.info("Dumping output")
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    print('ignoring final gulp')
                    continue # Ignore final gulp
        print("finished main")
                    


def main(argv):
    parser = argparse.ArgumentParser(description='LWA-SV ADP DRX Service')
    parser.add_argument('-f', '--fork',       action='store_true',       help='Fork and run in the background')
    parser.add_argument('-c', '--configfile', default='adp_config.json', help='Specify config file')
    parser.add_argument('-d', '--dryrun',     action='store_true',       help='Test without acting')
    parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
    parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
    parser.add_argument('-q', '--quiet',      action='count', default=0, help='Decrease verbosity')
    args = parser.parse_args()
    
    # Fork, if requested
    tuning = 0
    if args.fork:
        stderr = '/tmp/%s_%i.stderr' % (os.path.splitext(os.path.basename(__file__))[0], tuning)
        daemonize(stdin='/dev/null', stdout='/dev/null', stderr=stderr)
        
    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    if args.logfile is None:
        logHandler = logging.StreamHandler(sys.stdout)
    else:
        logHandler = Adp.AdpFileHandler(config, args.logfile)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    verbosity = args.verbose - args.quiet
    if   verbosity >  0: log.setLevel(logging.DEBUG)
    elif verbosity == 0: log.setLevel(logging.INFO)
    elif verbosity <  0: log.setLevel(logging.WARNING)
    

    short_date = ' '.join(__date__.split()[1:4])
    log.info("Starting %s with PID %i", argv[0], os.getpid())
    log.info("Cmdline args: \"%s\"", ' '.join(argv[1:]))
    log.info("Version:      %s", __version__)
    log.info("Last changed: %s", short_date)
    log.info("Config file:  %s", args.configfile)
    log.info("Log file:     %s", args.logfile)
    log.info("Dry run:      %r", args.dryrun)
    
    ops = []
    shutdown_event = threading.Event()
    def handle_signal_terminate(signum, frame):
        SIGNAL_NAMES = dict((k, v) for v, k in \
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and \
                            not v.startswith('SIG_'))
        log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
        try:
            ops[0].shutdown()
        except IndexError:
            pass
        shutdown_event.set()
    for sig in [signal.SIGHUP,
                signal.SIGINT,
                signal.SIGQUIT,
                signal.SIGTERM,
                signal.SIGTSTP]:
        signal.signal(sig, handle_signal_terminate)
    
    
    hostname = socket.gethostname()
    server_idx = 0 # HACK to allow testing on head node "adp"
    log.info("Hostname:     %s", hostname)
    log.info("Server index: %i", server_idx)
    
    iaddr = Address('0.0.0.0', 10000)
    isock = UDPSocket()
    isock.bind(iaddr)
    isock.timeout = 0.5

    print("binding input to", iaddr)
    
    capture_ring = Ring(name="capture-%i" % tuning, space='cuda_host')
    
    # TODO:  Figure out what to do with this resize
    GSIZE = 4096

    cores = list(range(8))
    
    nroach = 11
    nfreqblocks = 2
    roach0 = 0
    ops.append(CaptureOp(log, fmt="snap2", sock=isock, ring=capture_ring,
                         nsrc=nroach*nfreqblocks, src0=roach0, max_payload_size=9000,
                         buffer_ntime=GSIZE, slot_ntime=25000, core=cores.pop(0),
                         utc_start=datetime.datetime.now()))

    ops.append(DummyOp(log=log, iring=capture_ring, 
                            core=cores.pop(0), ntime_gulp=GSIZE))
        
    threads = [threading.Thread(target=op.main) for op in ops]
    
    log.info("Launching %i thread(s)", len(threads))
    for thread in threads:
        #thread.daemon = True
        thread.start()
    while not shutdown_event.is_set():
        signal.pause()
    log.info("Shutdown, waiting for threads to join")
    for thread in threads:
        thread.join()
    log.info("All done")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
