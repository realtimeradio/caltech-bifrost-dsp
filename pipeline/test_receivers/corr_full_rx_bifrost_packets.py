#!/usr/bin/env python

import os
import socket
import time
import struct
import sys
import argparse
import numpy as np
import json

#struct corr_output_full_packet {
#        uint64_t  sync_time;  // Unix time of last sync
#        uint64_t  second_count; // The _first_ spectrum in this integration (i.e., spectra since sync_time)
#        double    bw_hz;      // Hz bandwidth in this packet
#        double    sfreq;      // Center freq (in Hz) of first chan in this packet
#        uint32_t  acc_len;    // Accumulation length
#        uint32_t  nchans;     // Number of channels
#        uint32_t  chan0;      // First channel in this packet
#        uint32_t  npols;      // Number of polarizations
#        uint32_t  stand0;     // Stand 0
#        uint32_t  stand1;     // Stand 1
#        int32_t   data[npols, npols, nchans, 2]; // Data. __Little Endian__
#};

# For npols = 2; nchans=192; data payload is 6144 Bytes. Total packet size is 6184 Bytes

HEADER_SIZE = 32

def decode_header(p):
    x = struct.unpack('>IIIHHQIHH', p[0:HEADER_SIZE])
    rv = {}
    rv['sync_time'] = x[0]
    rv['frame_count'] = x[1]
    rv['second_count'] = x[2]
    rv['chan0'] = x[3]
    rv['gain'] = x[4]
    rv['time_tag'] = x[5]
    rv['navg'] = x[6]
    rv['stand0'] = x[7]
    rv['stand1'] = x[8]
    return rv

NPOL=2
NCHAN=96
NSTAND=352

parser = argparse.ArgumentParser(description='Receive X-Engine Full Correlator packets'
                                             'and write data to a file',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-P', '--port', type=int, default=11111,
                    help='UDP port to which to listen')
parser.add_argument('-i', '--ip', type=str, default='100.100.101.101',
                    help='IP address to which to bind')
parser.add_argument('-f', '--outpath', type=str, default='.',
                    help='Output file path')
parser.add_argument('-c', '--npipeline', type=int, default=8,
                    help='Number of pipelines expected')
parser.add_argument('-p', '--npol', type=int, default=NPOL,
                    help='Number of polarizations expected')
parser.add_argument('-s', '--nstand', type=int, default=NSTAND,
                    help='Number of stands expected')
args = parser.parse_args()

print("Creating socket and binding to %s:%d" % (args.ip, args.port))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((args.ip, args.port))

print("Number of channels: %d" % NCHAN)
print("Number of polarizations: %d" % args.npol)
print("Number of stands: %d" % args.nstand)
n_bl = args.nstand*(args.nstand+1) // 2
print("Number of baselines: %d" % n_bl)
n_pkt = n_bl * args.npipeline
payload_size = NCHAN * args.npol**2 * 2 * 4
packet_size = payload_size + HEADER_SIZE

time_tag = None
packet_cnt = 0
outbuf = np.zeros([args.nstand, args.nstand, args.npol, args.npol, 1024, 2], dtype=np.int32)

payload_dt = np.dtype(np.int32)
payload_dt = payload_dt.newbyteorder('>')
packet_buf = [b'\x00'*payload_size for _ in range(n_pkt * 2)]

for i in range(n_pkt * 2):
    packet_buf[i] = sock.recv(packet_size)

for p in packet_buf:
    h = decode_header(p)
    if h['time_tag'] != time_tag:
        time_tag = h['time_tag']
        print("New spectra ID: %d. Last ID had %d packets" % (time_tag, packet_cnt))
        packet_cnt = 0
    payload = np.frombuffer(p[HEADER_SIZE:], dtype=payload_dt).reshape([NCHAN, args.npol, args.npol, 2])
    for p0 in range(args.npol):
        for p1 in range(args.npol):
            outbuf[h['stand0']-1, h['stand1']-1, p0, p1, h['chan0']%1024:(h['chan0']%1024)+NCHAN, 0] = payload[:,p0,p1,0]
            outbuf[h['stand0']-1, h['stand1']-1, p0, p1, h['chan0']%1024:(h['chan0']%1024)+NCHAN, 1] = payload[:,p0,p1,1]
            outbuf[h['stand1']-1, h['stand0']-1, p0, p1, h['chan0']%1024:(h['chan0']%1024)+NCHAN, 0] = payload[:,p1,p0,0]
            outbuf[h['stand1']-1, h['stand0']-1, p0, p1, h['chan0']%1024:(h['chan0']%1024)+NCHAN, 1] = -payload[:,p1,p0,1]
    packet_cnt += 1
    if h['stand0'] == 1:
        if h['stand1'] == 1:
            print(h)
    if packet_cnt == n_pkt:
        print("Got %d packets" % packet_cnt)
        filename = "test_corr_full_rx_%dt_%dc_%dnc_%da.dat" % (
                    h['time_tag'],
                    h['chan0'],
                    NCHAN,
                    h['navg'],
                    )
        print("Writing %s" % filename)
        print("Acc len: %d" % h['navg'])
        print("Start channels : %d" % h['chan0'])
        out_meta = {}
        out_meta['ntime'] = 1
        out_meta['time'] = time.time()
        out_meta['nchan'] = NCHAN
        out_meta['chan0'] = h['chan0']
        out_meta['acc_len'] = h['navg']
        out_meta['t0'] = h['second_count']
        out_meta['type'] = 'corr_full_rx'
        out_meta['shape'] = outbuf.shape
        out_meta['dtype'] = str(outbuf.dtype)
        out_meta['nstand'] = args.nstand
        out_meta['npol'] = args.npol
        with open(os.path.join(args.outpath, filename), "wb") as fh:
            #fh.write(json.dumps(out_meta).encode())
            #fh.write('\n'.encode())
            fh.write(outbuf.tobytes())
        break
