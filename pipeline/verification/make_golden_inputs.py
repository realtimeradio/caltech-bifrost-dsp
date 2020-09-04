#!/usr/bin/env python

import numpy as np
import argparse
import os
import sys
import time

ACCSHORT = 2400
ACCLONG = 2400 * 100

NTIME = ACCLONG * 2
NSTAND = 352
NPOL = 2
NCHAN = 192

DATAPATH="/data/"

SEED = 0xdeadbeef

def main(argv):
    parser = argparse.ArgumentParser(description='Script for generating golden input / output files',
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--ntime',       type=int, default=NTIME,  help='Number of time samples for which data should be generated')
    parser.add_argument('-c', '--nchan',       type=int, default=NCHAN,  help='Number of freq channels for which data should be generated')
    parser.add_argument('-s', '--nstand',      type=int, default=NSTAND, help='Number of stands for which data should be generated')
    parser.add_argument('-p', '--npol',        type=int, default=NPOL,   help='Number of polarizations for which data should be generated')
    parser.add_argument('--accshort',          type=int, default=ACCSHORT, help='Number of samples to accumulate for fast correlations')
    parser.add_argument('--seed',              type=int, default=SEED,     help='Seed for random number generation')
    parser.add_argument('--nocorr',            action='store_true',        help='Do not generate correlation files')
    parser.add_argument('--datapath',          type=str, default=DATAPATH, help='Directory in which to put data')
    parser.add_argument('--chanramp',          action='store_true',        help='Make all test vectors a ramp with channel number')
    args = parser.parse_args()

    if args.nocorr:
        print("**NOT** writing correlation output files")

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
    if not args.nocorr:
        print('Total GBs corr output: %.2f' % (args.nstand**2 * args.npol**2 * args.nchan * args.ntime / args.accshort * 2 * 8 / 1e9))

    print()
    print('Seeding generator with %d' % args.seed)
    np.random.seed(args.seed)

    # Force generation in blocks of ACCSHORT samples, so we don't need too much memory
    assert (args.ntime % args.accshort == 0), "Number of samples to generate must be a multiple of fast accumulation length"

    assert (os.path.isdir(args.datapath)), "Data output directory %s does not exist!" % args.datapath

    if args.chanramp:
        in_outputfile = os.path.join(args.datapath, "in_%dt_%dc_%ds_%dp_chanramp.dat" % (args.ntime, args.nchan, args.nstand, args.npol))
        corr_outputfile = os.path.join(args.datapath, "corr_%dt_%da_%dc_%ds_%dp_chanramp.dat" % (args.ntime, args.accshort, args.nchan, args.nstand, args.npol))
    else:
        in_outputfile = os.path.join(args.datapath, "in_%dt_%dc_%ds_%dp_%x.dat" % (args.ntime, args.nchan, args.nstand, args.npol, args.seed))
        corr_outputfile = os.path.join(args.datapath, "corr_%dt_%da_%dc_%ds_%dp_%x.dat" % (args.ntime, args.accshort, args.nchan, args.nstand, args.npol, args.seed))

    nblock = args.ntime // args.accshort

    print()

    if not args.nocorr:
        corr_fh = open(corr_outputfile, 'wb')

    with open(in_outputfile, 'wb') as in_fh:
        now = time.time()

        if args.chanramp:
            d = np.zeros([args.nchan, args.nstand, args.npol], dtype=np.uint8)
            for s in range(args.nstand):
                for p in range(args.npol):
                    d[:,s,p] = np.arange(args.nchan, dtype=np.uint32) & 0xff
            # generate correlation
            dr = np.array(d >> 4, dtype=np.int8)
            dr[dr>7] -= 16
            di = np.array(d & 0xf, dtype=np.int8)
            di[di>7] -= 16
            dc = np.array(dr + 1j*di, dtype=np.complex)
            corr_out = np.zeros([args.nchan, args.nstand, args.nstand, args.npol, args.npol], dtype=np.complex)
            for p0 in range(args.npol):
                for p1 in range(args.npol):
                    corr_out[:,:,:,p0,p1] = (dc[:,:,p0,None] * np.conj(dc[:,None,:,p1]))*args.accshort
            for i in range(nblock):
                if time.time() - now > 5:
                    print("Generating block %d of %d" % (i+1, nblock))
                    now = time.time()
                for t in range(args.accshort):
                    in_fh.write(d.tobytes())
                if not args.nocorr:
                    corr_fh.write(corr_out.tobytes())
        else:
            for i in range(nblock):
                d = np.random.randint(0, 255, [args.accshort, args.nchan, args.nstand, args.npol], dtype=np.uint8)
                in_fh.write(d.tobytes())
                if time.time() - now > 10:
                    print("Generating block %d of %d" % (i+1, nblock))
                    now = time.time()

                if not args.nocorr:
                    # generate correlation
                    dr = np.array(d >> 4, dtype=np.int8)
                    dr[dr>7] -= 16
                    di = np.array(d & 0xf, dtype=np.int8)
                    di[di>7] -= 16
                    dc = np.array(dr + 1j*di, dtype=np.complex)
                    corr_out = np.zeros([args.nchan, args.nstand, args.nstand, args.npol, args.npol], dtype=np.complex)
                    for t in range(args.accshort):
                        # print status ~every 10 seconds
                        if time.time() - now > 10:
                            print("Generating time %d of %d for block %d of %d" % (t+1, args.accshort, i+1, nblock))
                            now = time.time()
                        for p0 in range(args.npol):
                            for p1 in range(args.npol):
                                corr_out[:,:,:,p0,p1] += (dc[t,:,:,p0,None] * np.conj(dc[t,:,None,:,p1]))
                    corr_fh.write(corr_out.tobytes())
        print("Closing input file")
    if not args.nocorr:
        print("Closing correlation file")
        corr_fh.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
