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
#        uint64_t  spectra_id; // The _first_ spectrum in this integration (i.e., spectra since sync_time)
#        uint32_t  acc_len;    // Accumulation length
#        uint32_t  chan0;   // First channel in this packet
#        uint32_t  npols;      // Number of polarizations
#        uint32_t  nchans;     // Number of channels
#        uint32_t  stand0;     // Stand 0
#        uint32_t  stand1;     // Stand 1
#        int32_t   data[npols, npols, nchans, 2]; // Data. __Little Endian__
#};

# For npols = 2; nchans=192; data payload is 6144 Bytes. Total packet size is 6184 Bytes

HEADER_SIZE = 40

def decode_header(p):
    x = struct.unpack('>QQ6L', p[0:HEADER_SIZE])
    rv = {}
    rv['sync_time'] = x[0]
    rv['spectra_id'] = x[1]
    rv['acc_len'] = x[2]
    rv['chan0'] = x[3]
    rv['npols'] = x[4]
    rv['nchans'] = x[5]
    rv['stand0'] = x[6]
    rv['stand1'] = x[7]
    return rv

NPOL=2
NCHAN=192
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
parser.add_argument('-c', '--nchan', type=int, default=NCHAN,
                    help='Number of freq channels expected')
parser.add_argument('-p', '--npol', type=int, default=NPOL,
                    help='Number of polarizations expected')
parser.add_argument('-s', '--nstand', type=int, default=NSTAND,
                    help='Number of stands expected')
args = parser.parse_args()

print("Creating socket and binding to %s:%d" % (args.ip, args.port))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((args.ip, args.port))

print("Number of channels: %d" % args.nchan)
print("Number of polarizations: %d" % args.npol)
print("Number of stands: %d" % args.nstand)
n_bl = args.nstand*(args.nstand+1) // 2
print("Number of baselines: %d" % n_bl)
payload_size = args.nchan * args.npol**2 * 2 * 4
packet_size = payload_size + HEADER_SIZE

spectra_id = None
packet_cnt = 0
outbuf = np.zeros([args.nstand, args.nstand, args.npol, args.npol, args.nchan], dtype=np.complex)

while(True):
    p = sock.recv(packet_size)
    h = decode_header(p)
    if h['spectra_id'] != spectra_id:
        spectra_id = h['spectra_id']
        print("New spectra ID: %d. Last ID had %d packets" % (spectra_id, packet_cnt))
        packet_cnt = 0
    payload = np.frombuffer(p[HEADER_SIZE:], dtype=np.int32).reshape([args.npol, args.npol, args.nchan, 2])
    outbuf[h['stand0'], h['stand1']].real = payload[:,:,:,0]
    outbuf[h['stand1'], h['stand0']].real = payload[:,:,:,0]
    outbuf[h['stand0'], h['stand1']].imag = payload[:,:,:,1]
    outbuf[h['stand1'], h['stand0']].imag = -payload[:,:,:,1]
    packet_cnt += 1
    if packet_cnt == n_bl:
        print("Got %d packets" % n_bl)
        filename = "test_corr_full_rx_%dt_%dc_%dnc_%da.dat" % (
                    h['spectra_id'],
                    h['chan0'],
                    h['nchans'],
                    h['acc_len'],
                    )
        print("Writing %s" % filename)
        print("spectra ID: %d" % h['spectra_id'])
        print("Acc len: %d" % h['acc_len'])
        print("Number of channels : %d" % h['nchans'])
        print("Start channels : %d" % h['chan0'])
        out_meta = {}
        out_meta['ntime'] = 1
        out_meta['time'] = time.time()
        out_meta['nchan'] = h['nchans']
        out_meta['chan0'] = h['chan0']
        out_meta['acc_len'] = h['acc_len']
        out_meta['t0'] = h['spectra_id']
        out_meta['type'] = 'corr_full_rx'
        out_meta['shape'] = outbuf.shape
        out_meta['dtype'] = str(outbuf.dtype)
        out_meta['nstand'] = args.nstand
        out_meta['npol'] = args.npol
        with open(os.path.join(args.outpath, filename), "wb") as fh:
            fh.write(json.dumps(out_meta).encode())
            fh.write('\n'.encode())
            fh.write(outbuf.tobytes())
        break
