#!/usr/bin/env python

import os
import socket
import time
import struct
import sys
import argparse
import numpy as np
import json

PACKET_BUF = 8192
BASE_HEADER_SIZE = 32

def decode_header(p):
    x = struct.unpack('>QIHHHHIII', p[0:BASE_HEADER_SIZE])
    h = {}
    h['spec_id'] = x[0]
    h['sync_time'] = x[1]
    h['n_pol'] = x[2]
    h['n_pol_tot'] = x[3]
    h['n_chan'] = x[4]
    h['n_chan_tot'] = x[5]
    h['chan_block'] = x[6]
    h['chan0'] = x[7]
    h['ant'] = x[8]
    payload_bytes = h['n_pol'] * h['n_chan']
    d = np.array(struct.unpack('>%dB' % payload_bytes, p[BASE_HEADER_SIZE:]), dtype=int)
    d_r = d >> 4
    d_r[d_r > 7] -= 16
    d_i = d & 0xf
    d_i[d_i > 7] -= 16
    data = d_r + 1j*d_i
    data = d
    return h, data.reshape(h['n_chan'], h['n_pol'])

parser = argparse.ArgumentParser(description='Receive F-Engine packets',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-P', '--port', type=int, default=11112,
                    help='UDP port to which to listen')
parser.add_argument('-i', '--ip', type=str, default='100.100.101.101',
                    help='IP address to which to bind')
parser.add_argument('-d', '--data', action='store_true',
                    help='Use this flag to print packet data rather than just headers')
args = parser.parse_args()

print("Creating socket and binding to %s:%d" % (args.ip, args.port))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((args.ip, args.port))

packet_cnt = 0
try:
    while(True):
        p = sock.recv(PACKET_BUF)
        h, data = decode_header(p)
        print(h)
        if args.data:
            print(data[0, :])
            for i in range(h['n_pol']):
                print(data[0:10, i])
        packet_cnt += 1
except KeyboardInterrupt:
    pass
