#!/usr/bin/env python

import os
import socket
import time
import struct
import sys
import argparse
import numpy as np
import json

#struct __attribute__((packed)) pbeam_hdr_type {
#        uint8_t  server;   // Note: 1-based
#        uint8_t  beam;     // Note: 1-based
#        uint8_t  gbe;      // (AKA tuning)
#        uint8_t  nchan;    // 109
#        uint8_t  nbeam;    // 2
#        uint8_t  nserver;  // 6
#        // Note: Big endian
#        uint16_t navg;     // Number of raw spectra averaged
#        uint16_t chan0;    // First chan in packet
#        uint64_t seq;      // Note: 1-based
#};

HEADER_SIZE = 18

def decode_header(p):
    x = struct.unpack('>BBBBBBHHQ', p[0:HEADER_SIZE])
    rv = {}
    rv['server'] = x[0]
    rv['beam'] = x[1]
    rv['tuning'] = x[2]
    rv['nchan'] = x[3]
    rv['nbeam'] = x[4]
    rv['nserver'] = x[5]
    rv['navg'] = x[6]
    rv['chan0'] = x[7]
    rv['seq'] = x[8]
    return rv

def decode_data(p, nchan, nbeam=1):
    NPOL = 2
    nwords = nchan * nbeam * NPOL
    d = struct.unpack('<%df' % (nwords*2), p[HEADER_SIZE:])
    dr = np.array(d[0::2])
    di = np.array(d[1::2])
    return dr + 1j*di



parser = argparse.ArgumentParser(description='Receive P-Beam packets',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-P', '--port', type=int, default=11111,
                    help='UDP port to which to listen')
parser.add_argument('-i', '--ip', type=str, default='100.100.101.101',
                    help='IP address to which to bind')
args = parser.parse_args()

print("Creating socket and binding to %s:%d" % (args.ip, args.port))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((args.ip, args.port))

time_tag = None
packet_cnt = 0
seq = 0
last_seq=0

while(True):
    p = sock.recv(8192)
    packet_cnt += 1
    h = decode_header(p)
    d = decode_data(p, h['nchan'], nbeam=h['nbeam'])
    seq = h['seq']
    if not np.all(d == 0):
        seqdelta = seq-last_seq
        dumpdelta = seqdelta / h['navg']
        print(h, packet_cnt, seqdelta, dumpdelta)
        last_seq = seq
        #print(d.shape, d[0:10])
    else:
        continue
        print(d.shape, "ALL ZERO")
