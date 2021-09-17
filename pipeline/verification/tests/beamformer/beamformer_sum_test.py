#! /usr/bin/env python

"""
Write random data and weights (not using the control interface)
to the bifrost Beamform block.

Verify that Beamform->BeamformSumBeams produces
results consistent with a simple python implementation.
"""

import sys
import os
import time
import logging
import threading
import ujson as json
import numpy as np
import etcd3 as etcd

from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
from lwa352_pipeline.blocks.dummy_source_block import DummySource
from lwa352_pipeline.blocks.copy_block import Copy
from lwa352_pipeline.blocks.beamform_block import Beamform
from lwa352_pipeline.blocks.beamform_sum_beams_block import BeamformSumBeams
from lwa352_pipeline.blocks.block_base import Block

NTIME_SUM = 24
NTIME_GULP = NTIME_SUM*5 # 480
NSTAND = 704
NPOL = 2
NGULP = 7 # Number of gulps of random data to generate
NCHAN = 16 # 96
GPUDEV = 0
ETCD_HOST = "etcdv3service.sas.pvt"
TESTFILE_NAME = '/tmp/temp_beam_testfile.dat'
NBEAM = 16

CORES = list(range(12))

class SoftwareBfSum(Block):
    def __init__(self, log, iring, ntime_gulp=2500,
            guarantee=True, core=-1, testfile=None):

        super(SoftwareBfSum, self).__init__(log, iring, None, guarantee, core, etcd_client=None)
        cpu_affinity.set_core(self.core)
        self.ntime_gulp = ntime_gulp
        # file containing test data
        if testfile is not None:
            self.testfile = open(testfile, 'rb')
            self.testfile_nbytes = os.path.getsize(testfile)
        else:
            self.testfile = None

    def get_input_gulp(self, t, acc_len, nbeam, nchan):
        # data file has order chan x beam x time x complexity
        nbytes = self.ntime_gulp * acc_len * 2*nbeam * nchan * 2 * 4
        seekloc = (t*nbytes) % self.testfile_nbytes
        self.testfile.seek(seekloc)
        rawdata = self.testfile.read(nbytes)
        data = np.frombuffer(rawdata, dtype=np.complex64).reshape(nchan, 2*nbeam, self.ntime_gulp*acc_len)
        return data

    def cpu_sum_power(self, data, ntime_sum):
        # data: [chan x beam x time]
        nchan, nbeam, ntime = data.shape
        out = np.zeros([nbeam//2, ntime // ntime_sum, nchan, 4], dtype=np.float32)
        for t in range(ntime // ntime_sum):
            #print("Summing time %d" % t)
            d = data[:,:,t*ntime_sum:(t+1)*ntime_sum]
            for b in range(nbeam//2):
                out[b, t, :, 0] = np.sum(np.abs(d[:,2*b,:])**2, axis=1)
                out[b, t, :, 1] = np.sum(np.abs(d[:,2*b+1,:])**2, axis=1)
                cross = np.sum(d[:,2*b,:] * np.conj(d[:,2*b+1,:]), axis=1)
                out[b, t, :, 2] = cross.real
                out[b, t, :, 3] = cross.imag
        return out

    def main(self):
        cpu_affinity.set_core(self.core)

        tick = time.time()
        for iseq in self.iring.read(guarantee=self.guarantee):
            gulpi = 0
            ihdr = json.loads(iseq.header.tobytes())
            nstand = ihdr['nstand']
            npol = ihdr['npol']
            nbeam = ihdr['nbeam']
            nchan = ihdr['nchan']
            acc_len = ihdr['acc_len']
            seq = ihdr['seq0']
            igulp_size = self.ntime_gulp * nchan * nbeam * 4 * 4 # 4*float32
            running_max = 0
            for ispan in iseq.read(igulp_size):
                idata = ispan.data_view('f32').reshape([nbeam, self.ntime_gulp, nchan, 4])
                test_data = self.get_input_gulp(seq // self.ntime_gulp // acc_len, acc_len, nbeam, nchan)
                cpudata = self.cpu_sum_power(test_data, acc_len)
                #print(idata.shape, test_data.shape, cpudata.shape)
                #print(idata[0,0,0:10])
                #print(cpudata[0,0,0:10])
                maxdiff = np.max(np.abs(idata - cpudata)/np.abs(idata))
                if maxdiff > running_max:
                    running_max = maxdiff
                assert np.all(np.isclose(idata, cpudata, atol=1e-4)), "CPU/GPU match fail! Max frac diff %f" % maxdiff
                seq += self.ntime_gulp*acc_len
                print("MATCH OK after %d times (max frac diff %f)" % (seq, running_max))
                #if time.time() - tick > 1:
                #    print('Gulp: %d' % gulpi)
                #    print('Time (beam0, chan0):', idata[0, 0, 0:10])
                #    print('Chan (time 0, beam 0):', idata[:,0,0])
                #    print('Beam (time 0, chan0):', idata[0,:,0])
                #    tick = time.time()
                gulpi += 1

def main():
    rng = np.random.default_rng(0xaabbccdd)
    print("Generating test data")
    test_data_r = rng.random(size=[NGULP, NCHAN, NBEAM*2, NTIME_GULP]) - 0.5
    test_data_i = rng.random(size=[NGULP, NCHAN, NBEAM*2, NTIME_GULP]) - 0.5
    test_data = np.array(test_data_r + 1j*test_data_i, dtype=np.complex64) * 5

    print("Writing test data to file")
    with open(TESTFILE_NAME, 'wb') as fh:
        print("Writing testfile: %s" % TESTFILE_NAME)
        fh.write(test_data.tobytes())
        print("Closing file")

    etcd_client = None #etcd.client(ETCD_HOST)

    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)

    input_ring = Ring(name="input", space="cuda_host")
    gpu_input_ring = Ring(name="gpu_input", space="cuda")
    bf_output_ring = Ring(name="bf_output", space="cuda")
    output_ring = Ring(name="output", space="cuda_host")

    ops = []
    # Up nstand to 8*NBEAM to account for the fact that the dummy block is configured
    # for 4+4 bit data, and we want to send 32+32 bit
    ops.append(DummySource(log, oring=input_ring, ntime_gulp=NTIME_GULP,
                   core=CORES.pop(0), skip_write=False,
                   nstand=8*NBEAM, nchan=NCHAN, npol=2, testfile=TESTFILE_NAME, header={'nbeam':NBEAM*2}))
    
    ops.append(Copy(log, iring=input_ring, oring=gpu_input_ring, ntime_gulp=NTIME_GULP,
                      nbyte_per_time=8*NBEAM*NCHAN*2,
                      core=CORES.pop(0), guarantee=True, gpu=GPUDEV))

    # Instantiate a Beamform block as this initialized the beamforming library
    Beamform(log, iring=gpu_input_ring, oring=bf_output_ring, ntime_gulp=NTIME_GULP,
                  nchan=NCHAN, nbeam=NBEAM*2, ninput=NSTAND*NPOL,
                  core=CORES[0], guarantee=True, gpu=GPUDEV, ntime_sum=None,
                  etcd_client=etcd_client)
    
    ops.append(BeamformSumBeams(log, iring=gpu_input_ring, oring=bf_output_ring, ntime_gulp=NTIME_GULP,
                      nchan=NCHAN, core=CORES.pop(0), guarantee=True, gpu=GPUDEV, ntime_sum=NTIME_SUM))

    ops.append(Copy(log, iring=bf_output_ring, oring=output_ring, ntime_gulp=NTIME_GULP//NTIME_SUM,
                      nbyte_per_time=NBEAM*NCHAN*4*4,
                      core=CORES.pop(0), guarantee=True, gpu=GPUDEV))

    ops.append(SoftwareBfSum(log, iring=output_ring, ntime_gulp=NTIME_GULP//NTIME_SUM,
        testfile=TESTFILE_NAME))

    threads = [threading.Thread(target=op.main) for op in ops]
    
    log.info("Launching %i thread(s)", len(threads))
    try:
        for thread in threads:
            thread.start()
        while (True):
            try:
                pass
            except KeyboardInterrupt:
                log.info("Shutdown, waiting for threads to join")
                for thread in threads:
                    thread.join()
                break
        log.info("All done")
        return 0
    except:
        raise

if __name__ == '__main__':
    main()
