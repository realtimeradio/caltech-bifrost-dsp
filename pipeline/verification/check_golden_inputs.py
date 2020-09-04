#!/usr/bin/env python

import numpy as np
import argparse
import os
import sys
import time

ACCSHORT = 2400
ACCLONG = 2400 * 100

NTIME = ACCLONG * 4
NSTAND = 352
NPOL = 2
NCHAN = 192

DATAPATH="/data/"

SEED = 0xdeadbeef

def main(argv):
    parser = argparse.ArgumentParser(description='Script for generating golden input / output files',
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--infile',       type=str, default=None,  help='File containing input timestream data')
    parser.add_argument('-o', '--corrfile',     type=str, default=None,  help='File containing partially accumulated correlation data')
    parser.add_argument('-t', '--ntime',       type=int, default=NTIME,  help='Number of time samples for which data should be generated')
    parser.add_argument('-c', '--nchan',       type=int, default=NCHAN,  help='Number of freq channels for which data should be generated')
    parser.add_argument('-s', '--nstand',      type=int, default=NSTAND, help='Number of stands for which data should be generated')
    parser.add_argument('-p', '--npol',        type=int, default=NPOL,   help='Number of polarizations for which data should be generated')
    parser.add_argument('--accshort',          type=int, default=ACCSHORT, help='Number of samples to accumulate for fast correlations')
    parser.add_argument('--seed',              type=int, default=SEED,     help='Seed for random number generation')
    parser.add_argument('--datapath',          type=str, default=DATAPATH, help='Directory in which to put data')
    args = parser.parse_args()

    print("times:", args.ntime)
    print("chans:", args.nchan)
    print("stands:", args.nstand)
    print("pols:", args.npol)

    print()
    print("Short accumulation length", args.accshort)

    nval = args.ntime * args.nchan * args.nstand * args.npol

    print()
    print('Total values:', nval)
    print('Total GBs input: %.2f' % (nval / 1e9))
    print('Total GBs corr output: %.2f' % (args.nstand**2 * args.npol**2 * args.nchan * args.ntime / args.accshort * 2 * 8 / 1e9))

    print()
    print('Seeding generator with %d' % args.seed)
    np.random.seed(args.seed)

    # Force generation in blocks of ACCSHORT samples, so we don't need too much memory
    assert (args.ntime % args.accshort == 0), "Number of samples to generate must be a multiple of fast accumulation length"

    assert (os.path.isdir(args.datapath)), "Data output directory %s does not exist!" % args.datapath

    nblock = args.ntime // args.accshort

    print()
    match = True
    with open(args.infile, 'rb') as in_fh:
        with open(args.corrfile, 'rb') as corr_fh:
            now = time.time()
            for i in range(nblock):
                # Read a block of time data
                d_raw = in_fh.read(args.accshort * args.nchan * args.nstand * args.npol)
                d = np.frombuffer(d_raw, dtype=np.uint8).reshape([args.accshort, args.nchan, args.nstand, args.npol])
                corr_raw = corr_fh.read(args.nchan * args.nstand * args.nstand * args.npol * args.npol * 2 * 8)
                corr = np.frombuffer(corr_raw, dtype=np.complex).reshape([args.nchan, args.nstand, args.nstand, args.npol, args.npol])
                # compute corr in numbest way possible and verify
                dr = np.array(d >> 4, dtype=np.int8)
                dr[dr>7] -= 16
                di = np.array(d & 0xf, dtype=np.int8)
                di[di>7] -= 16
                dc = np.array(dr + 1j*di, dtype=np.complex)
                print('computing corr')
                for c in range(args.nchan):
                    for s0 in range(args.nstand):
                        for s1 in range(args.nstand):
                            if time.time() - 10 > now:
                                print("Processing time block %d of %d, chan %d of %d, baseline (%d,%d)" % (i+1, nblock, c+1, args.nchan, s0, s1))
                                now = time.time()
                            for p0 in range(args.npol):
                                for p1 in range(args.npol):
                                    x = 0
                                    for t in range(args.accshort):
                                        x += dc[t, c, s0, p0] * np.conj(dc[t, c, s1, p1])
                                if x != corr[c, s0, s1, p0, p1]:
                                    match = False
                                    print("Stand %d Pol %d * Stand %d Pol %d match?" % (s0, p0, s1, p1), x==corr[c, s0, s1, p0, p1])
                                    print(x)
                                    print(corr[c,s0,s1,p0,p1])

    print("Vectors match?", match)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
