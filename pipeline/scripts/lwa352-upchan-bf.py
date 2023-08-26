#! /usr/bin/env python

import signal
import logging
import time
import os
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

# Yukky hacks to crowbar in pipeline indices recorder expects
NSERVER = 8
NPIPELINE_PER_SERVER = 4

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
    import bifrost as bf
    import bifrost.blocks as blocks
    import bifrost.views as views
    from bifrost.address import Address
    from bifrost.udp_socket import UDPSocket
    from bifrost.ring import Ring
    # Blocks
    from lwa352_pipeline.blocks.block_base import Block
    from lwa352_pipeline.blocks.trigger_source_block import TrigBufSourceBlock
    from lwa352_pipeline.blocks.beamform_offline_block import BfOfflineBlock

    # Set the pipeline ID
    Block.set_id(args.pipelineid)

    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    if args.logfile is None:
        logHandler = logging.StreamHandler(sys.stdout)
    else:
        logHandler = logging.FileHandler(args.logfile)
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
    log.info("Log file:     %s", args.logfile)
    
    ops = []
    shutdown_event = threading.Event()
    
    hostname = socket.gethostname()
    log.info("Hostname:     %s", hostname)
    log.info("Pipeline index: %i", args.pipelineid)

    GSIZE = 480 # Number of time samples to read from file in one go
    NUPCHAN = 32

    assert GSIZE % NUPCHAN == 0, "Gulp size must be a multiple of upchannelization factor"

    cores = CoreList(map(int, args.cores.split(',')))

    # Open a file and write GSIZE time samples to the input_ring buffer
    raw_data = TrigBufSourceBlock([args.datain], gulp_nframe=GSIZE//NUPCHAN, frame_size=NUPCHAN)
    gpu_raw_data = blocks.copy(raw_data, space='cuda')
    transposed_data = blocks.transpose(gpu_raw_data, ['time', 'freq', 'stand', 'pol', 'fine_time'])
    upchan_data = blocks.fft(transposed_data, axes='fine_time', axis_labels='fine_freq')

    ra_array = list(map(float, args.ra_array.split(',')))
    dec_array = list(map(float, args.dec_array.split(',')))
    bf_data = BfOfflineBlock(upchan_data, args.nbeam, args.nbeams_per_batch, args.ntimestep, ra_array, dec_array)

    pipeline = bf.get_default_pipeline()
    pipeline.shutdown_on_signals()
    print(pipeline.dot_graph())
    pipeline.run()


def main(argv):
    parser = argparse.ArgumentParser(description='LWA352-OVRO Correlator-Beamformer Pipeline',
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
    parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
    parser.add_argument('--nchan',            type=int, default=96,      help='Number of frequency channels in the pipeline')
    parser.add_argument('--datain',           type=str, default=None,    help='Path to input data file')
    parser.add_argument('-G', '--gpu',        type=int, default=0,       help='Which GPU device to use')
    parser.add_argument('-P', '--pipelineid', type=int, default=0,       help='Pipeline ID. Useful if you are running multiple pipelines on a single machine')
    parser.add_argument('-C', '--cores',      default='0,1,2,3,4,5,6,7', help='Comma-separated list of CPU cores to use')
    parser.add_argument('-q', '--quiet',      action='count', default=0, help='Decrease verbosity')
    parser.add_argument('--target_throughput', type=float, default='1000.0',  help='Target throughput when using --fakesource')
    parser.add_argument('--nbeam', type=int, default=4, help='Number of beams to form')
    parser.add_argument('--nbeams_per_batch', type=int, default=2, help='Number of beams per batch')
    parser.add_argument('--ntimestep', type=int, default=1024, help='Number of time samples between coefficient updates')
    parser.add_argument('--ra_array', type=str, default="10,20,30,40", help='Comma-separated values for RA array')
    parser.add_argument('--dec_array', type=str, default="10,20,30,40", help='Comma-separated values for DEC array')

    args = parser.parse_args()

    build_pipeline(args)
    

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
