import socket
import time
import struct
import sys

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


seq = 0
magic = 0xaabbccdd
npol = 64
nchan = 96
chan0 = 0
pol0 = 0

npol_total = 704
nchan_total = nchan*2

nchan_blocks = nchan_total // nchan
npol_blocks = npol_total // npol

nwords = npol * nchan

payload = struct.pack('>%dB' % nwords, *[0 for _ in range(nwords)])

seq_stat_period = 3e3
tick = 0
while(True):
    for chan_block_id in range(nchan_blocks):
       for pol in range(npol_blocks):
           header = struct.pack('>QLHHHHLLL', seq, magic, npol, npol_total, nchan, nchan_total, chan_block_id, chan_block_id*nchan, pol)
           data = header + payload
           sock.sendto(data, (sys.argv[1], 10000))
           #time.sleep(0.00001)
    seq += 1
    if seq % seq_stat_period == 0:
        tock = time.time()
        dt = tock - tick
        tick = time.time()
        mbytes = seq_stat_period * len(payload) * npol_blocks * nchan_blocks / 1e6
        print("Dumped 1M packets %d MBytes (%.2f MB/s)" % (mbytes, mbytes/dt))
