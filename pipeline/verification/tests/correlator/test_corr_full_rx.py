#!/usr/bin/env python
"""
Test an integration received via the python corr_full_test_rx.py
script -- which writes an nstand x nstand x nchan x npol x nchan x 2 array
of np.int32 -- against the correlation golden vector files.
"""

import numpy as np
import argparse
import os
import sys
import time
import json

def main(argv):
    parser = argparse.ArgumentParser(description='Script for generating golden input / output files',
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--goldfile',    type=str, default=None,  help='File containing golden corr data')
    parser.add_argument('-u', '--uutfile',    type=str, default=None,  help='File containing data from the real correlator')
    args = parser.parse_args()

    with open(args.goldfile, 'rb') as fh:
        gold_header = json.loads(fh.readline().decode())
        gold_data = fh.read()

    with open(args.uutfile, 'rb') as fh:
        uut_header = json.loads(fh.readline().decode())
        uut_data = fh.read()

    # recode known datatypes
    gold_header['dtype'] = gold_header['dtype'].lstrip('np.')
    uut_header['dtype'] = uut_header['dtype'].lstrip('np.')

    print("UUT header:", uut_header)
    print("Gold header:", gold_header)

    gold_array = np.frombuffer(gold_data, dtype=gold_header['dtype']).reshape(gold_header['shape'])
    uut_array = np.frombuffer(uut_data, dtype=uut_header['dtype']).reshape(uut_header['shape'])

    print("Gold shape:", gold_array.shape)
    print("UUT shape:", uut_array.shape)

    # To save time we can construct the UUT accumulation by multiplying up
    # the gold vectors.
    assert uut_header['ntime'] * uut_header['acc_len'] % gold_header['acc_len'] == 0
    assert uut_header['t0'] % gold_header['acc_len'] == 0
    gold_repeats = uut_header['ntime'] * uut_header['acc_len'] // (gold_header['acc_len'] * gold_header['ntime'])
    print("Complete repeats of gold file:", gold_repeats)
    gold_extra_accs = (uut_header['ntime'] * uut_header['acc_len'] // gold_header['acc_len']) % gold_header['ntime']
    gold_start_time = (uut_header['t0'] // gold_header['acc_len']) % (gold_header['ntime'])
    print("Extra integrations of gold file: %d, starting at %d" % (gold_extra_accs, gold_start_time))

    gold_accumulator = np.zeros(gold_array.shape[1:], dtype=gold_array.dtype)
    for t in range(gold_start_time, gold_start_time + gold_extra_accs):
        gold_accumulator += gold_array[t % gold_header['ntime']]
    gold_accumulator += gold_array.sum(axis=0) * gold_repeats

    print("Comparing gold and UUT")
    errs = 0
    ok = 0
    for s0 in range(uut_header['nstand']):
        for s1 in range(uut_header['nstand']):
            for p0 in range(uut_header['npol']):
                for p1 in range(uut_header['npol']):
                    if not np.all(gold_accumulator[:,s0,s1,p0,p1] == uut_array[s0,s1,p0,p1,:]):
                        print("Error! s0: %d, p0: %d, s1: %d, p1: %d" % (s0, p0, s1, p1)) 
                        print("gold:", gold_accumulator[0:5,s0,s1,p0,p1])
                        print("uut:", uut_array[s0,s1,p0,p1,0:5])
                        errs += 1
                    else:
                        ok += 1

    if errs == 0:
        print("PASSED, with no errors (%d OK)" % ok)
    else:
        print("FAILED, with %d errors (%d OK)" % (errs, ok))

if __name__ == '__main__':
    sys.exit(main(sys.argv))
