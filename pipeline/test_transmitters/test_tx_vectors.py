import socket
import time
import struct
import sys
import argparse

NPOL = 2
NCHAN = 192
NSTAND = 352

seq = 0
magic = 0xaabbccdd
nchan_per_pkt = 64
nstand_per_pkt = 32
chan0 = 0
pol0 = 0

parser = argparse.ArgumentParser(description='Emulate a bunch of SNAP2s using input'
                                             'data from a test file',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-P', '--port', type=int, default=10000,
                    help='UDP port to which data should be sent')
parser.add_argument('-i', '--ip', type=str, default='100.100.100.100',
                    help='IP address to which data should be sent')
parser.add_argument('-f', '--testfile', type=str, default=None,
                    help='File containing test data')
parser.add_argument('-c', '--nchan', type=int, default=NCHAN,
                    help='Number of freq channels for which data should be generated')
parser.add_argument('-s', '--nstand', type=int, default=NSTAND,
                    help='Number of stands for which data should be generated')
parser.add_argument('-p', '--npol', type=int, default=NPOL,
                    help='Number of polarizations for which data should be generated')
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#struct snap2_hdr_type {
#        uint64_t  seq;       // Spectra counter == packet counter
#        uint32_t  magic;     // = 0xaabbccdd
#        uint16_t  npol;      // Number of polarizations in this packet
#        uint16_t  npol_tot;      // Total number of polarizations for this pipeline
#        uint16_t  nchan;     // Number of channels in this packet
#        uint16_t  nchan_tot;     // Total number of channels for this pipeline
#        uint32_t  chan_block_id; // Channel block ID. Eg. 0 for chans 0..nchan-1, 1 for chans nchan..2*nchan-1, etc.
#        uint32_t  chan0;     // First channel in this packet 
#        uint32_t  pol0;      // First pol in this packet 
#};


# input data has format NCHAN * NSTAND * NPOL

nchan_blocks = args.nchan // nchan_per_pkt
npol_blocks = args.nstand // nstand_per_pkt

nbytes = args.npol * nstand_per_pkt * nchan_per_pkt

seq_stat_period = 3e3
tick = 0
with open(args.testfile, 'rb') as fh:
    while(True):
        try:
            for chan_block_id in range(nchan_blocks):
               for pol_block_id in range(npol_blocks):
                   header = struct.pack('>QLHHHHLLL', seq, magic,
                              args.npol*nstand_per_pkt,
                              args.nstand*args.npol,
                              nchan_per_pkt, args.nchan,
                              chan_block_id, chan_block_id*nchan_per_pkt,
                              pol_block_id * nstand_per_pkt * args.npol)
                   payload = fh.read(nbytes)
                   if len(payload) == 0:
                       fh.seek(0)
                   payload = fh.read(nbytes)
                   data = header + payload
                   sock.sendto(data, (args.ip, args.port))
                   #time.sleep(0.00001)
            seq += 1
            if seq % seq_stat_period == 0:
                tock = time.time()
                dt = tock - tick
                tick = time.time()
                mbytes = seq_stat_period * len(payload) * npol_blocks * nchan_blocks / 1e6
                print("Dumped 1M packets %d MBytes (%.2f MB/s)" % (mbytes, mbytes/dt))
        except KeyboardInterrupt:
            break
