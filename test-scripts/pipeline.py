from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture
from bifrost.packet_writer import HeaderInfo, DiskWriter, UDPTransmit
from bifrost.ring import Ring, WriteSpan
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

from bifrost.libbifrost import _bf

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
import numpy as np

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

class DummySource(object):
    """
    A dummy source block for throughput testing. Does nothing
    but mark input buffers ready for consumption.
    """
    def __init__(self, log, oring, ntime_gulp=2500,
                 core=-1, nchans=192, npols=704):
        self.log = log
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.core = core
        self.nchans = nchans
        self.npols = npols
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.gulp_size = self.ntime_gulp*nchans*npols*1        # complex8

        self.test_data = 1*np.ones(self.gulp_size, dtype=np.uint8)

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        time.sleep(0.1)
        self.oring.resize(self.gulp_size)
        hdr = {}
        hdr['nchans'] = self.nchans
        hdr['npols'] = self.npols
        hdr['seq0'] = 0
        time_tag = 0
        REPORT_PERIOD = 100
        bytes_per_report = REPORT_PERIOD * self.gulp_size
        with self.oring.begin_writing() as oring:
            tick = time.time()
            while(True):
                ohdr_str = json.dumps(hdr)
                with oring.begin_sequence(time_tag=time_tag, header=ohdr_str) as oseq:
                    with oseq.reserve(self.gulp_size) as ospan:
                        ospan.data[...] = self.test_data
                        time_tag += 1
                        hdr['seq0'] += self.ntime_gulp
                if time_tag % REPORT_PERIOD == 0:
                    tock = time.time()
                    dt = tock - tick
                    print('Send %d bytes in %.2f seconds (%.2f Gb/s)' % (bytes_per_report, dt, (8*bytes_per_report / dt / 1e9)))
                    tick = tock

class CopyOp(object):
    """
    Copy data from one buffer to another.
    """
    def __init__(self, log, iring, oring, ntime_gulp=2500,
                 guarantee=True, core=-1, nchans=192, npols=704):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.guarantee = guarantee
        self.core = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nchans*npols*1        # complex8

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.igulp_size)
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet) as oseq:
                    for ispan in iseq.read(self.igulp_size):
                        with oseq.reserve(self.igulp_size) as ospan:
                            #self.log.debug("Copying output")
                            odata = ospan.data_view('ci4')
                            copy_array(ospan.data, ispan.data)

class Corr(object):
    """
    Perform cross-correlation using xGPU
    """
    def __init__(self, log, iring, oring, ntime_gulp=2500,
                 guarantee=True, core=-1, nchans=192, npols=704, acc_len=2400):
        assert (acc_len % ntime_gulp == 0), "Acculmulation length must be a multiple of gulp size"
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.guarantee = guarantee
        self.core = core
        self.acc_len = acc_len
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nchans*npols*1        # complex8
        self.ogulp_size = 47849472 * 8 # complex64

        # initialize xGPU. Arrays passed as inputs don't really matter here
        # but we need to pass something
        ibuf = BFArray(0, dtype='i8', space='cuda')
        obuf = BFArray(0, dtype='i64', space='cuda')
        rv = _bf.xgpuInitialize(ibuf.as_BFarray(), obuf.as_BFarray(), 0)
        if (rv != _bf.BF_STATUS_SUCCESS):
            self.log.error("xgpuIntialize returned %d" % rv)
            raise RuntimeError

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                self.log.debug("Correlating output")
                ihdr = json.loads(iseq.header.tostring())
                subacc_id = ihdr['seq0'] % self.acc_len
                first = subacc_id == 0
                last = subacc_id == self.acc_len - self.ntime_gulp
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                for ispan in iseq.read(self.igulp_size):
                    if first:
                        oseq = oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                    _bf.xgpuKernel(ispan.data.as_BFarray(), ospan.data.as_BFarray(), int(last))
                    if last:
                        ospan.close()
                        oseq.end()

class CorrSubSel(object):
    """
    Grab arbitrary entries from a GPU buffer and copy them to the CPU
    """
    def __init__(self, log, iring, oring,
            guarantee=True, core=-1, nchans=192):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.guarantee = guarantee
        self.core = core
        
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
        self.obuf_gpu = BFArray(shape=[nchans, self.nvis_out], dtype='i64', space='cuda')
        self.ogulp_size = len(self.subsel) * 8

    def main(self):
        cpu_affinity.set_core(self.core)
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
                    self.log.debug("GRabbing subselection")
                    idata = ispan.data_view('i64').reshape(47849472)
                    with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet) as oseq:
                        with oseq.reserve(self.ogulp_size) as ospan:
                            rv = _bf.xgpuSubSelect(idata.as_BFarray(), self.obuf_gpu.as_BFarray(), self.subsel.as_BFarray())
                            if (rv != _bf.BF_STATUS_SUCCESS):
                                self.log.error("xgpuIntialize returned %d" % rv)
                                raise RuntimeError
                            #copy_array(ospan.data, obuf_gpu.data)

class CorrAcc(object):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring, oring, ntime_gulp=2400,
                 guarantee=True, core=-1, nchans=192, npols=704, acc_len=24000):
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        self.log = log
        self.iring = iring
        self.oring = oring
        self.guarantee = guarantee
        self.core = core
        self.ntime_gulp = ntime_gulp
        self.acc_len = acc_len
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = 47849472 * 8 # complex64
        self.ogulp_size = self.igulp_size
        # integration buffer
        self.accdata = BFArray(shape=(self.igulp_size // 4), dtype='i32', space='cuda')
        self.bfbf = LinAlg()

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        oseq = None
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                subacc_id = ihdr['seq0'] % self.acc_len
                first = subacc_id == 0
                last = subacc_id == self.acc_len - self.ntime_gulp
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                for ispan in iseq.read(self.igulp_size):
                    self.log.debug("Accumulating correlation")
                    idata = ispan.data_view('i32')
                    if first:
                        oseq = oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                        # TODO: surely there are more sensible ways to implement a vacc
                        BFMap("a = b", data={'a': self.accdata, 'b': idata})
                    else:
                        BFMap("a += b", data={'a': self.accdata, 'b': idata})
                    if last:
                        if oseq is None:
                            print("Skipping output because oseq isn't open")
                        else:
                            # copy to CPU
                            odata = ospan.data_view('i32')
                            odata = self.accdata
                            print(odata[0:10])
                            ospan.close()
                            oseq.end()
                            oseq = None

class CorrSub(object):
    """
    Subselect entries from a full visitibility matrix and copy these
    from GPU to CPU
    """
    def __init__(self, log, iring, oring, ntime_gulp=2500,
                 guarantee=True, core=-1, nchans=192, npols=704):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.guarantee = guarantee
        self.core = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nchans*npols*1        # complex8
        self.ogulp_size = 47849472 * 8 # complex64

        # initialize xGPU. Arrays passed as inputs don't really matter here
        # but passing something prevents xGPU from trying to allocate
        # host memory
        ibuf = BFArray(self.igulp_size, dtype='ci4')
        obuf = BFArray(self.ogulp_size // 8, dtype='ci32')
        _bf.xgpuInitialize(ibuf.as_BFarray(), obuf.as_BFarray(), 0)

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        #with self.oring.begin_writing() as oring:
        #    for iseq in self.iring.read(guarantee=self.guarantee):
        #        self.log.info("Correlating output")
        #        ihdr = json.loads(iseq.header.tostring())
        #        subacc_id = ihdr['seq0'] % self.acc_len
        #        first = subacc_id == 0
        #        last = subacc_id == self.acc_len - self.ntime_gulp
        #        ohdr = ihdr.copy()
        #        # Mash header in here if you want
        #        ohdr_str = json.dumps(ohdr)
        #        for ispan in iseq.read(self.igulp_size):
        #            if first:
        #                oseq = oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet)
        #                ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
        #            _bf.xgpuCorrelate(ispan.data.as_BFarray(), ospan.data.as_BFarray(), int(last))
        #            if last:
        #                ospan.close()
        #                oseq.end()

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
        for iseq in self.iring.read(guarantee=True):
            self.log.debug("Dumping output")
            #for ispan in iseq.read(igulp_size):
            #    if ispan.size < igulp_size:
            #        print('ignoring final gulp')
            #        continue # Ignore final gulp
                    


def main(argv):
    parser = argparse.ArgumentParser(description='LWA-SV ADP DRX Service')
    parser.add_argument('-f', '--fork',       action='store_true',       help='Fork and run in the background')
    parser.add_argument('-c', '--configfile', default='adp_config.json', help='Specify config file')
    parser.add_argument('-d', '--dryrun',     action='store_true',       help='Test without acting')
    parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
    parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
    parser.add_argument('--fakesource',       action='store_true',       help='Use a dummy source for testing')
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
    
    capture_ring = Ring(name="capture", space='cuda_host')
    gpu_input_ring = Ring(name="gpu-input", space='cuda')
    corr_output_ring = Ring(name="corr-output", space='cuda')
    corr_slow_output_ring = Ring(name="corr-slow-output", space='cuda_host')
    corr_fast_output_ring = Ring(name="corr-fast-output", space='cuda_host')
    
    # TODO:  Figure out what to do with this resize
    GSIZE = 480#1200
    SLOT_NTIME = 2*GSIZE

    cores = list(range(8))
    
    nroach = 11
    nfreqblocks = 2
    roach0 = 0
    if not args.fakesource:
        print("binding input to", iaddr)
        iaddr = Address('100.100.100.101', 10000)
        isock = UDPSocket()
        isock.bind(iaddr)
        isock.timeout = 0.5
        ops.append(CaptureOp(log, fmt="snap2", sock=isock, ring=capture_ring,
                             nsrc=nroach*nfreqblocks, src0=0, max_payload_size=9000,
                             buffer_ntime=GSIZE, slot_ntime=SLOT_NTIME, core=cores.pop(0),
                             utc_start=datetime.datetime.now()))
    else:
        print('Using dummy source...')
        ops.append(DummySource(log, oring=capture_ring, ntime_gulp=GSIZE, core=cores.pop(0)))

    ## capture_ring -> triggered buffer

    ops.append(CopyOp(log, iring=capture_ring, oring=gpu_input_ring, ntime_gulp=GSIZE,
                      core=cores.pop(0), guarantee=True))

    ## gpu_input_ring -> beamformer
    ## beamformer -> UDP

    ops.append(Corr(log, iring=gpu_input_ring, oring=corr_output_ring, ntime_gulp=GSIZE,
                      core=cores.pop(0), guarantee=True, acc_len=2400))

    ops.append(CorrSubSel(log, iring=corr_output_ring, oring=corr_fast_output_ring,
                      core=cores.pop(0), guarantee=True))

    ops.append(CorrAcc(log, iring=corr_output_ring, oring=corr_slow_output_ring,
                      core=cores.pop(0), guarantee=True, acc_len=24000))
    #
    ## corr_slow_output -> UDP
    ## corr_fast_output -> UDP

    final_ring = corr_fast_output_ring
    #final_ring = corr_output_ring

    ops.append(DummyOp(log=log, iring=final_ring, 
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
