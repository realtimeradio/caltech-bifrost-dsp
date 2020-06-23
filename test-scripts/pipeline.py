from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.ring import Ring


import signal
import logging
import time
import os
import argparse
import threading
import socket
import datetime

# Blocks
from blocks.corr_block import Corr
from blocks.dummy_source_block import DummySource
from blocks.corr_acc_block import CorrAcc
from blocks.corr_subsel_block import CorrSubSel
from blocks.copy_block import Copy
from blocks.capture_block import Capture
from blocks.beamform_block import Beamform

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

def main(argv):
    parser = argparse.ArgumentParser(description='LWA-SV ADP DRX Service')
    parser.add_argument('-f', '--fork',       action='store_true',       help='Fork and run in the background')
    parser.add_argument('-c', '--configfile', default='adp_config.json', help='Specify config file')
    parser.add_argument('-d', '--dryrun',     action='store_true',       help='Test without acting')
    parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
    parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
    parser.add_argument('--fakesource',       action='store_true',       help='Use a dummy source for testing')
    parser.add_argument('--nodata',           action='store_true',       help='Don\'t generate data in the dummy source (faster)')
    parser.add_argument('--nocorr',           action='store_true',       help='Don\'t use correlation threads')
    parser.add_argument('--nobeamform',       action='store_true',       help='Don\'t use beamforming threads')
    parser.add_argument('--nogpu',            action='store_true',       help='Don\'t use any GPU threads')
    parser.add_argument('-G', '--gpu',        type=int, default=0,       help='Which GPU device to use')
    parser.add_argument('-C', '--cores',      default='0,1,2,3,4,5,6,7', help='Comma-separated list of CPU cores to use')
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
    bf_output_ring = Ring(name="bf-output", space='cuda')
    corr_output_ring = Ring(name="corr-output", space='cuda')
    corr_slow_output_ring = Ring(name="corr-slow-output", space='cuda_host')
    corr_fast_output_ring = Ring(name="corr-fast-output", space='cuda_host')
    
    # TODO:  Figure out what to do with this resize
    GSIZE = 480#1200
    SLOT_NTIME = 2*GSIZE # What does this do? JD says maybe nothing :)
    nstand = 352
    npol = 2
    nchans = 192

    cores = list(map(int, args.cores.split(',')))
    
    nroach = 11
    nfreqblocks = 2
    roach0 = 0
    if not args.fakesource:
        ip_str = '100.100.100.101'
        print("binding input to", ip_str)
        iaddr = Address(ip_str, 10000)
        isock = UDPSocket()
        isock.bind(iaddr)
        isock.timeout = 0.5
        ops.append(Capture(log, fmt="snap2", sock=isock, ring=capture_ring,
                           nsrc=nroach*nfreqblocks, src0=0, max_payload_size=9000,
                           buffer_ntime=GSIZE, slot_ntime=SLOT_NTIME, core=cores.pop(0),
                           utc_start=datetime.datetime.now()))
    else:
        print('Using dummy source...')
        ops.append(DummySource(log, oring=capture_ring, ntime_gulp=GSIZE, core=cores.pop(0), skip_write=args.nodata))

    ## capture_ring -> triggered buffer

    if not args.nogpu:
        ops.append(Copy(log, iring=capture_ring, oring=gpu_input_ring, ntime_gulp=GSIZE,
                          core=cores.pop(0), guarantee=True, gpu=args.gpu))

    if not (args.nobeamform or args.nogpu):
        ops.append(Beamform(log, iring=gpu_input_ring, oring=bf_output_ring, ntime_gulp=GSIZE,
                          nchan_max=nchans, nbeam_max=1, nstand=nstand, npol=npol,
                          core=cores.pop(0), guarantee=True, gpu=args.gpu))

    ## gpu_input_ring -> beamformer
    ## beamformer -> UDP

    if not (args.nocorr or args.nogpu):
        ops.append(Corr(log, iring=gpu_input_ring, oring=corr_output_ring, ntime_gulp=GSIZE,
                          core=cores.pop(0), guarantee=True, acc_len=2400, gpu=args.gpu))

        ops.append(CorrSubSel(log, iring=corr_output_ring, oring=corr_fast_output_ring,
                          core=cores.pop(0), guarantee=True, gpu=args.gpu))

        ops.append(CorrAcc(log, iring=corr_output_ring, oring=corr_slow_output_ring,
                          core=cores.pop(0), guarantee=True, acc_len=24000, gpu=args.gpu))
    #
    ## corr_slow_output -> UDP
    ## corr_fast_output -> UDP

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
