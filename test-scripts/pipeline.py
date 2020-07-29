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
from blocks.beamform_sum_block import BeamformSum
from blocks.beamform_vlbi_block import BeamformVlbi
from blocks.beamform_vacc_block import BeamVacc

import etcd3 as etcd

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
    parser = argparse.ArgumentParser(description='LWA-SV ADP DRX Service',
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--ibverbs',          action='store_true',       help='Use IB verbs for packet capture')
    parser.add_argument('-G', '--gpu',        type=int, default=0,       help='Which GPU device to use')
    parser.add_argument('-C', '--cores',      default='0,1,2,3,4,5,6,7', help='Comma-separated list of CPU cores to use')
    parser.add_argument('-q', '--quiet',      action='count', default=0, help='Decrease verbosity')
    parser.add_argument('--testcorr',         action='store_true',       help='Compare the GPU correlation with CPU. SLOW!!')
    parser.add_argument('--useetcd',          action='store_true',       help='Use etcd control/monitoring server')
    parser.add_argument('--etcdhost',         default='etcdhost',        help='Host serving etcd functionality')
    parser.add_argument('--ip',               default='100.100.100.101', help='IP address to which to bind')
    parser.add_argument('--target_throughput', type=float, default='1000.0',  help='Target throughput when using --fakesource')
    args = parser.parse_args()
    
    if args.useetcd:
        etcd_client = etcd.client(args.etcdhost)
    else:
        etcd_client = None

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
    
    NBEAM = 16
    if not args.nogpu:
        capture_ring = Ring(name="capture", space='cuda_host')
        gpu_input_ring = Ring(name="gpu-input", space='cuda')
        bf_output_ring = Ring(name="bf-output", space='cuda')
        bf_power_output_ring = Ring(name="bf-output", space='cuda_host')
        bf_vlbi_output_ring = Ring(name="bf-output", space='cuda_host')
        bf_acc_output_ring = [Ring(name="bf-output", space='system') for _ in range(NBEAM)]
        corr_output_ring = Ring(name="corr-output", space='cuda')
        corr_slow_output_ring = Ring(name="corr-slow-output", space='cuda_host')
        corr_fast_output_ring = Ring(name="corr-fast-output", space='cuda_host')
    else:
        capture_ring = Ring(name="capture", space='system')
    
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
        print("binding input to", args.ip)
        iaddr = Address(args.ip, 10000)
        isock = UDPSocket()
        isock.bind(iaddr)
        isock.timeout = 0.5
        ops.append(Capture(log, fmt="snap2", sock=isock, ring=capture_ring,
                           nsrc=nroach*nfreqblocks, src0=0, max_payload_size=9000,
                           buffer_ntime=GSIZE, slot_ntime=SLOT_NTIME, core=cores.pop(0),
                           utc_start=datetime.datetime.now(), ibverbs=args.ibverbs))
    else:
        print('Using dummy source...')
        ops.append(DummySource(log, oring=capture_ring, ntime_gulp=GSIZE, core=cores.pop(0), skip_write=args.nodata, target_throughput=args.target_throughput))

    ## capture_ring -> triggered buffer

    if not args.nogpu:
        ops.append(Copy(log, iring=capture_ring, oring=gpu_input_ring, ntime_gulp=GSIZE,
                          core=cores.pop(0), guarantee=True, gpu=args.gpu))

    if not (args.nocorr or args.nogpu):
        ops.append(Corr(log, iring=gpu_input_ring, oring=corr_output_ring, ntime_gulp=GSIZE,
                          core=cores.pop(0), guarantee=True, acc_len=2400, gpu=args.gpu, test=args.testcorr, etcd_client=etcd_client))

        ops.append(CorrSubSel(log, iring=corr_output_ring, oring=corr_fast_output_ring,
                          core=cores.pop(0), guarantee=True, gpu=args.gpu, etcd_client=etcd_client))

        ops.append(CorrAcc(log, iring=corr_output_ring, oring=corr_slow_output_ring,
                          core=cores.pop(0), guarantee=True, acc_len=24000, gpu=args.gpu))

    if not (args.nobeamform or args.nogpu):
        ops.append(Beamform(log, iring=gpu_input_ring, oring=bf_output_ring, ntime_gulp=GSIZE,
                          nchan_max=nchans, nbeam_max=NBEAM*2, nstand=nstand, npol=npol,
                          core=cores.pop(0), guarantee=True, gpu=args.gpu, ntime_sum=None))
        ops.append(BeamformSum(log, iring=bf_output_ring, oring=bf_power_output_ring, ntime_gulp=GSIZE,
                          nchan_max=nchans, nbeam_max=NBEAM*2, nstand=nstand, npol=npol,
                          core=cores.pop(0), guarantee=True, gpu=args.gpu, ntime_sum=24))
        ops.append(BeamformVlbi(log, iring=bf_output_ring, oring=bf_vlbi_output_ring, ntime_gulp=GSIZE,
                          nchan_max=nchans, ninput_beam=NBEAM, npol=npol,
                          core=cores.pop(0), guarantee=True, gpu=args.gpu))
        for i in range(1):
            ops.append(BeamVacc(log, iring=bf_power_output_ring, oring=bf_acc_output_ring[i], nint=GSIZE//24, beam_id=i,
                          nchans=nchans, ninput_beam=NBEAM,
                          core=cores.pop(0), guarantee=True))

    ## gpu_input_ring -> beamformer
    ## beamformer -> UDP

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
