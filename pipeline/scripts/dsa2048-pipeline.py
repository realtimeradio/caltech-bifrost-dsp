#! /usr/bin/env python

import signal
import logging
import time
import os
import sys
import argparse
import threading
import socket
import datetime

__version__    = "1.0"
__date__       = '$LastChangedDate: 2020-25-11$'
__author__     = "Jack Hickish, Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = ""
__credits__    = ["Jack Hickish", "Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jack Hickish"
__email__      = "jack@realtimeradio.co.uk"
__status__     = "Development"

class CoreList(list):
    """
    A dumb class to catch pop-ing too many cores and print an error
    """
    def pop(self, i):
        try:
            return list.pop(self, i)
        except IndexError:
            print("Ran out of CPU cores to use!")
            exit()

def build_pipeline(args):
    print('Importing libraries...', end='')
    sys.stdout.flush()
    from bifrost.address import Address
    from bifrost.udp_socket import UDPSocket
    from bifrost.ring import Ring
    # Blocks
    from lwa352_pipeline.blocks.block_base import Block
    from lwa352_pipeline.blocks.corr_block import Corr
    from lwa352_pipeline.blocks.dummy_source_block import DummySource
    from lwa352_pipeline.blocks.corr_acc_block import CorrAcc
    from lwa352_pipeline.blocks.corr_subsel_block import CorrSubsel
    from lwa352_pipeline.blocks.corr_output_full_block import CorrOutputFull
    from lwa352_pipeline.blocks.corr_output_part_block import CorrOutputPart
    from lwa352_pipeline.blocks.copy_block import Copy
    from lwa352_pipeline.blocks.capture_block import Capture
    from lwa352_pipeline.blocks.beamform_block import Beamform
    from lwa352_pipeline.blocks.beamform_sum_beams_block import BeamformSumBeams
    from lwa352_pipeline.blocks.beamform_vlbi_output_block import BeamformVlbiOutput
    from lwa352_pipeline.blocks.beamform_output_block import BeamformOutput
    from lwa352_pipeline.blocks.triggered_dump_block import TriggeredDump

    from lwa352_pipeline.blocks.romein import RomeinNoFFT
    print('Done')

    if args.useetcd:
        import etcd3 as etcd
        etcd_client = etcd.client(args.etcdhost)
    else:
        etcd_client = None

    # Set the pipeline ID
    Block.set_id(args.pipelineid)

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
    log.info("Starting %s with PID %i", sys.argv[0], os.getpid())
    log.info("Cmdline args: \"%s\"", ' '.join(sys.argv[1:]))
    log.info("Version:      %s", __version__)
    log.info("Last changed: %s", short_date)
    log.info("Config file:  %s", args.configfile)
    log.info("Log file:     %s", args.logfile)
    
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
        corr_output_ring = Ring(name="corr-output", space='cuda')
        grid_output_ring = Ring(name="grid-output", space='cuda_host')
    else:
        capture_ring = Ring(name="capture", space='system')

    
    # TODO:  Figure out what to do with this resize
    #GSIZE = 480#1200
    GSIZE = 512
    nstand = 2048
    npol = 2
    nchan = 32
    system_nchan = 450 * 32
    CORR_SUBSEL_NCHAN_SUM = 4 # Number of freq chans to average over while sub-selecting baselines

    cores = CoreList(map(int, args.cores.split(',')))
    
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
                           nsrc=nroach*nfreqblocks, src0=0, max_payload_size=7000,
                           buffer_ntime=GSIZE, core=cores.pop(0), system_nchan=system_nchan,
                           utc_start=datetime.datetime.now(), ibverbs=args.ibverbs))
    else:
        print('Using dummy source...')
        ops.append(DummySource(log, oring=capture_ring, ntime_gulp=GSIZE, core=cores.pop(0),
                   skip_write=args.nodata, target_throughput=args.target_throughput,
                   nstand=nstand, nchan=nchan, npol=npol, testfile=args.testdatain))

    # Get the antenna to input map as understood by the data source
    # This could (should?) to passed down in the headers and calculated on the fly,
    # but observational evidence is that this can be problematic for pipeline throughput.
    ant_to_input = ops[-1].ant_to_input

    if not args.nogpu:
        ops.append(Copy(log, iring=capture_ring, oring=gpu_input_ring, ntime_gulp=GSIZE,
                          nbyte_per_time=nchan*npol*nstand,
                          core=cores.pop(0), guarantee=True, gpu=args.gpu))

    if not (args.nocorr or args.nogpu):
        ops.append(Corr(log, iring=gpu_input_ring, oring=corr_output_ring, ntime_gulp=GSIZE,
                          nchan=nchan, npol=npol, nstand=nstand,
                          core=cores.pop(0), guarantee=True, acc_len=512*256, gpu=args.gpu, test=args.testcorr, etcd_client=etcd_client, autostartat=512*8*8, ant_to_input=ant_to_input))

        ops.append(RomeinNoFFT(log, iring=corr_output_ring, oring=grid_output_ring,
                               conv=5, grid=4096, core=cores.pop(0), nant=nstand, gpu=args.gpu))

    threads = [threading.Thread(target=op.main) for op in ops]
    
    log.info("Launching %i thread(s)", len(threads))
    try:
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
    except:
        raise

def main(argv):
    parser = argparse.ArgumentParser(description='LWA352-OVRO Correlator-Beamformer Pipeline',
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--fork',       action='store_true',       help='Fork and run in the background')
    parser.add_argument('-c', '--configfile', default='adp_config.json', help='Specify config file')
    parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
    parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
    parser.add_argument('--fakesource',       action='store_true',       help='Use a dummy source for testing')
    parser.add_argument('--nodata',           action='store_true',       help='Don\'t generate data in the dummy source (faster)')
    parser.add_argument('--testdatain',       type=str, default=None,    help='Path to input test data file')
    parser.add_argument('--testdatacorr',     type=str, default=None,    help='Path to correlator output test data file')
    parser.add_argument('--testdatacorr_acc_len', type=int, default=2400, help='Number of accumulations per sample in correlator test data file')
    #parser.add_argument('-a', '--corr_acc_len',   type=int, default=240000, help='Number of accumulations to start accumulating in the slow correlator')
    parser.add_argument('--nocorr',           action='store_true',       help='Don\'t use correlation threads')
    #parser.add_argument('--nobeamform',       action='store_true',       help='Don\'t use beamforming threads')
    parser.add_argument('--nogpu',            action='store_true',       help='Don\'t use any GPU threads')
    parser.add_argument('--ibverbs',          action='store_true',       help='Use IB verbs for packet capture')
    parser.add_argument('-G', '--gpu',        type=int, default=0,       help='Which GPU device to use')
    parser.add_argument('-P', '--pipelineid', type=int, default=0,       help='Pipeline ID. Useful if you are running multiple pipelines on a single machine')
    parser.add_argument('-C', '--cores',      default='0,1,2,3,4,5,6,7', help='Comma-separated list of CPU cores to use')
    parser.add_argument('-q', '--quiet',      action='count', default=0, help='Decrease verbosity')
    parser.add_argument('--testcorr',         action='store_true',       help='Compare the GPU correlation with CPU. SLOW!!')
    parser.add_argument('--useetcd',          action='store_true',       help='Use etcd control/monitoring server')
    parser.add_argument('--etcdhost',         default='etcdhost',        help='Host serving etcd functionality')
    parser.add_argument('--ip',               default='100.100.100.101', help='IP address to which to bind')
    parser.add_argument('--bufgbytes',        type=int, default=4,       help='Number of GBytes to buffer for transient buffering')
    parser.add_argument('--target_throughput', type=float, default='1000.0',  help='Target throughput when using --fakesource')
    args = parser.parse_args()

    build_pipeline(args)
    

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
