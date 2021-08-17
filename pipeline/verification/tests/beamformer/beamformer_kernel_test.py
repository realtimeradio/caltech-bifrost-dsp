#! /usr/bin/env python

import sys
import time
import logging
import threading
import ujson as json
import numpy as np
import etcd3 as etcd

import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
from bifrost.ndarray import copy_array

NSTAND = 352
NPOL = 2
NTIME_GULP = 480
NCHAN = 96
GPUDEV = 0
ETCD_HOST = "etcdv3service.sas.pvt"
TESTFILE_NAME = '/tmp/temp_testfile.dat'
NBEAM = 16

CORES = list(range(12))

def main():
    #test_data = np.zeros([NTIME_GULP, NCHAN, NSTAND, NPOL], dtype=np.uint8)
    test_data = np.ones([NTIME_GULP, NCHAN, NSTAND*NPOL], dtype=np.uint8)
    for i in range(NTIME_GULP):
        test_data[i,:,:] = ((i%8) << 4)
    #for s in range(NSTAND):
    #    test_data[:, :, s, :] = NSTAND % 256
    #with open(TESTFILE_NAME, 'wb') as fh:
    #    fh.write(test_data.tobytes())

    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)

    gains = np.ones([NBEAM, NCHAN, NSTAND*NPOL], dtype=np.complex64)
    #gains = np.ones([NCHAN, NBEAM, NSTAND*NPOL], dtype=np.complex64)
    for i in range(NBEAM):
        gains[i,:,:] = i+1

    gains = BFArray(gains, dtype='cf32', space='cuda')
    idata = BFArray(test_data, dtype='u8', space='cuda')
    #odata = BFArray(shape=(NCHAN, NTIME_GULP, NBEAM), dtype='cf32', space='cuda')
    #odata_cpu = BFArray(shape=(NCHAN, NTIME_GULP, NBEAM), dtype='cf32', space='cuda_host')
    odata = BFArray(shape=(NCHAN, NBEAM, NTIME_GULP), dtype='cf32', space='cuda')
    odata_cpu = BFArray(shape=(NCHAN, NBEAM, NTIME_GULP), dtype='cf32', space='cuda_host')

    _bf.bfBeamformInitialize(GPUDEV, NSTAND*NPOL, NCHAN, NTIME_GULP, NBEAM, 0)
    _bf.bfBeamformRun(idata.as_BFarray(), odata.as_BFarray(), gains.as_BFarray())

    odata_cpu[...] = odata
    print(odata_cpu.shape)
    #print('TIME', odata_cpu[0,:,0])
    #print('CHAN', odata_cpu[:,0,0])
    #print('BEAM_c0_t0', odata_cpu[0,0,:])
    #print('BEAM_c1_t0', odata_cpu[1,0,:])
    #print('BEAM_c0_t1', odata_cpu[0,1,:])
    #print('BEAM_c1_t1', odata_cpu[1,1,:])
    print('TIME', odata_cpu[0,0,:])
    print('CHAN', odata_cpu[:,0,0])
    print('BEAM_c0_t0', odata_cpu[0,:,0])
    print('BEAM_c1_t0', odata_cpu[1,:,0])
    print('BEAM_c0_t1', odata_cpu[0,:,1])
    print('BEAM_c1_t1', odata_cpu[1,:,1])

if __name__ == '__main__':
    main()
