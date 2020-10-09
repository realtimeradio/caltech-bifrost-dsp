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
#        double    bw_hz;      // Hz bandwidth in this packet
#        double    sfreq;      // Center freq (in Hz) of first chan in this packet
#        uint32_t  acc_len;    // Accumulation length
#        uint32_t  nvis;       // Number of visibilities per packet
#        uint32_t  nchans;     // Number of channels
#        uint32_t  chan0;      // First channel in this packet
#        uint32_t  baselines[nvis, 2, 2] // baseline IDs
#        int32_t   data[nvis, nchans, 2]; // Data. __Big Endian__
#};


BASE_HEADER_SIZE = 48

def decode_header(p):
    x = struct.unpack('>QQ2d4I', p[0:BASE_HEADER_SIZE])
    rv = {}
    rv['sync_time'] = x[0]
    rv['spectra_id'] = x[1]
    rv['bw'] = x[2]
    rv['sfreq'] = x[3]
    rv['acc_len'] = x[4]
    rv['nvis'] = x[5]
    rv['nchans'] = x[6]
    rv['chan0'] = x[7]
    rv['baselines'] = struct.unpack('>%dI' % (4*rv['nvis']), p[BASE_HEADER_SIZE:BASE_HEADER_SIZE+4*4*rv['nvis']])
    return rv

NPOL=2
NCHAN=192 // 4
NSTAND=352
NBL = 4656
NBLPKT = 16

parser = argparse.ArgumentParser(description='Receive X-Engine Full Correlator packets'
                                             'and write data to a file',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-P', '--port', type=int, default=11112,
                    help='UDP port to which to listen')
parser.add_argument('-i', '--ip', type=str, default='100.100.101.101',
                    help='IP address to which to bind')
parser.add_argument('-f', '--outpath', type=str, default='.',
                    help='Output file path')
parser.add_argument('-c', '--nchan', type=int, default=NCHAN,
                    help='Number of freq channels expected')
parser.add_argument('-b', '--nbl', type=int, default=NBL,
                    help='Number of baselines expected')
parser.add_argument('-p', '--nblpkt', type=int, default=NBLPKT,
                    help='Number of baselines expected per packet')
args = parser.parse_args()

assert args.nbl % args.nblpkt == 0

print("Creating socket and binding to %s:%d" % (args.ip, args.port))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((args.ip, args.port))

print("Number of channels: %d" % args.nchan)
print("Number of baselines: %d" % args.nbl)
print("Number of baselines per packet: %d" % args.nblpkt)
payload_size = args.nchan * args.nblpkt * 2 * 4
packet_size = payload_size + BASE_HEADER_SIZE + 4*4*args.nblpkt

packet_cnt = 0
outbuf = np.zeros([args.nbl, args.nchan], dtype=np.complex)
outbls = np.zeros([args.nbl, 2, 2], dtype=np.int32)

payload_dt = np.dtype(np.int32)
payload_dt = payload_dt.newbyteorder('>')

blcnt = 0
wait = True
p = sock.recv(packet_size)
h = decode_header(p)
first_spectra_id = h['spectra_id']
while(True):
    p = sock.recv(packet_size)
    h = decode_header(p)
    # Spin until the spectra ID changes
    if wait:
        if h['spectra_id'] != first_spectra_id:
            wait = False
            target_spectra_id = h['spectra_id']
            print("New spectra ID: %d" % (target_spectra_id))
            packet_cnt = 0
        else:
            continue
    if h['spectra_id'] != target_spectra_id:
        print("SPECTRA ID MISMATCH!")
        exit()
    payload = np.frombuffer(p[BASE_HEADER_SIZE+4*4*h['nvis']:], dtype=payload_dt).reshape([args.nblpkt, args.nchan, 2])
    baselines = np.array(h['baselines']).reshape([args.nblpkt, 2, 2])
    for b in range(h['nvis']):
        outbls[blcnt + b] = baselines[b]
        outbuf[blcnt + b, :].real = payload[b,:,0]
        outbuf[blcnt + b, :].imag = payload[b,:,1]
    blcnt += args.nblpkt
    packet_cnt += 1
    if packet_cnt == args.nbl // args.nblpkt:
        print("Got %d baselines" % args.nbl)
        filename = "test_corr_part_rx_%dt_%dc_%dnc_%da.dat" % (
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
        out_meta['type'] = 'corr_part_rx'
        out_meta['shape'] = outbuf.shape
        out_meta['dtype'] = str(outbuf.dtype)
        out_meta['nbl'] = args.nbl
        out_meta['nblpkt'] = args.nblpkt
        out_meta['baselines'] = outbls.tolist()
        with open(os.path.join(args.outpath, filename), "wb") as fh:
            fh.write(json.dumps(out_meta).encode())
            fh.write('\n'.encode())
            fh.write(outbuf.tobytes())
        break
