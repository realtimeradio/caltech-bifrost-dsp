#! /usr/bin/env python

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
from lwa352_pipeline.blocks.block_base import Block

from lwa352_pipeline_control.etcd_control import EtcdCorrControl
from lwa352_pipeline_control.blocks.beamform_control import BeamformControl

NSTAND = 40 #352
NPOL = 2
NTIME_GULP = 120 # 480
NGULP = 5 # Number of gulps of random data to generate
NCHAN = 16 # 96
GPUDEV = 0
ETCD_HOST = "etcdv3service.sas.pvt"
TESTFILE_NAME = '/tmp/temp_testfile.dat'
NBEAM = 16

chan_bw_hz = 24e3

CORES = list(range(12))

class SoftwareBf(Block):
    def __init__(self, log, iring, ntime_gulp=2500,
            guarantee=True, core=-1, testfile=None, coeffs=None):

        super(SoftwareBf, self).__init__(log, iring, None, guarantee, core, etcd_client=None)
        cpu_affinity.set_core(self.core)
        self.ntime_gulp = ntime_gulp
        # file containing test data
        if testfile is not None:
            self.testfile = open(testfile, 'rb')
            self.testfile_nbytes = os.path.getsize(testfile)
        else:
            self.testfile = None

        self.coeffs = coeffs
        self.nchan, self.nbeam, self.ninput = self.coeffs.shape

    def get_input_gulp(self, t):
        nbytes = self.ntime_gulp * self.nchan * self.ninput
        seekloc = (t*nbytes) % self.testfile_nbytes
        self.testfile.seek(seekloc)
        rawdata = self.testfile.read(nbytes)
        uint_data = np.frombuffer(rawdata, dtype=np.uint8).reshape([self.ntime_gulp, self.nchan, self.ninput])
        complex_data = np.zeros_like(uint_data, dtype=np.complex64)
        r = np.array((uint_data >> 4) & 0xf, dtype=np.int8)
        r[r>7] -= 16
        i = np.array(uint_data & 0xf, dtype=np.int8)
        i[i>7] -= 16
        complex_data = r + 1j*i
        return complex_data

    def cpu_beamform(self, data):
        # data: [time x chan x input]
        # coeffs: [chan x beam x input]
        out = np.zeros([self.nchan, self.nbeam, self.ntime_gulp], dtype=np.complex64)
        for t in range(self.ntime_gulp):
            for c in range(self.nchan):
                for b in range(self.nbeam):
                    out[c,b,t] = np.sum(self.coeffs[c,b,:] * data[t,c,:])
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
            seq = ihdr['seq0']
            igulp_size = self.ntime_gulp * nchan * nbeam * 2 * 4 # complex float32
            running_max = 0
            for ispan in iseq.read(igulp_size):
                idata = ispan.data_view('cf32').reshape([nchan, nbeam, self.ntime_gulp])
                test_data = self.get_input_gulp(seq // self.ntime_gulp)
                cpudata = self.cpu_beamform(test_data)
                #print(idata[0,0,0:10])
                #print(cpudata[0,0,0:10])
                maxdiff = np.max(np.abs(idata - cpudata)/np.abs(idata))
                if maxdiff > running_max:
                    running_max = maxdiff
                assert np.all(np.isclose(idata, cpudata, rtol=1e-4, atol=1e-4)), "CPU/GPU match fail! Max frac diff %f" % maxdiff
                seq += self.ntime_gulp
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
    test_data = rng.integers(0, high=255, size=[NGULP*NTIME_GULP, NCHAN, NSTAND, NPOL], dtype=np.uint8)

    print("Writing test data to file")
    with open(TESTFILE_NAME, 'wb') as fh:
        print("Writing testfile: %s" % TESTFILE_NAME)
        fh.write(test_data.tobytes())

    print("Generating random coefficients")
    calgains = (3*rng.random(size=[NCHAN, 2*NBEAM, NSTAND*NPOL])+4) + 1j*(4*rng.random(size=[NCHAN, 2*NBEAM, NSTAND*NPOL])+5)
    calgains = np.array(calgains, dtype=np.complex64)
    beamdelays = 12*rng.random(size=[2*NBEAM, NSTAND*NPOL], dtype=np.float32)
    beamamps   = 7*rng.random(size=[2*NBEAM, NSTAND*NPOL], dtype=np.float32) + 10

    coeffs = np.zeros([NCHAN, 2*NBEAM, NSTAND*NPOL], dtype=np.complex64)
    for b in range(2*NBEAM):
        for i in range(NSTAND*NPOL):
            coeffs[:,b,i] = calgains[:,b,i] * beamamps[b,i] * np.exp(1j*2*np.pi*beamdelays[b,i]/1e9*chan_bw_hz*np.arange(NCHAN))
    
    etcd_client = etcd.client(ETCD_HOST)

    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)

    print("Generating simulated control objects")
    sim_etcd_control = EtcdCorrControl(simulated=True, log=log)
    sim_ctrl = BeamformControl(log, sim_etcd_control, '')

    input_ring = Ring(name="input", space="cuda_host")
    gpu_input_ring = Ring(name="gpu_input", space="cuda")
    bf_output_ring = Ring(name="bf_output", space="cuda")
    output_ring = Ring(name="output", space="cuda_host")

    ops = []
    ops.append(DummySource(log, oring=input_ring, ntime_gulp=NTIME_GULP,
                   core=CORES.pop(0), skip_write=False,
                   nstand=NSTAND, nchan=NCHAN, npol=NPOL, testfile=TESTFILE_NAME))
    
    ops.append(Copy(log, iring=input_ring, oring=gpu_input_ring, ntime_gulp=NTIME_GULP,
                      nbyte_per_time=NCHAN*NPOL*NSTAND,
                      core=CORES.pop(0), guarantee=True, gpu=GPUDEV))
    
    ops.append(Beamform(log, iring=gpu_input_ring, oring=bf_output_ring, ntime_gulp=NTIME_GULP,
                      nchan=NCHAN, nbeam=NBEAM*2, ninput=NSTAND*NPOL,
                      core=CORES.pop(0), guarantee=True, gpu=GPUDEV, ntime_sum=None,
                      etcd_client=etcd_client))

    ops[-1].freqs = chan_bw_hz*np.arange(ops[-1].nchan)
    print("Constucting weight-setting commands")
    commands = []
    for b in range(2*NBEAM):
        for i in range(NSTAND*NPOL):
            commands += [sim_ctrl.update_calibration_gains(b, i, calgains[:, b, i])]
    for b in range(2*NBEAM):
        commands += [sim_ctrl.update_delays(b, beamdelays[b], beamamps[b])]
    print("Processing commands")
    ops[-1].process_command_strings(commands)
    #ops[-1].gains_cpu[...] = coeffs

    ops.append(Copy(log, iring=bf_output_ring, oring=output_ring, ntime_gulp=NTIME_GULP,
                      nbyte_per_time=2*NBEAM*NCHAN*2*4,
                      core=CORES.pop(0), guarantee=True, gpu=GPUDEV))

    ops.append(SoftwareBf(log, iring=output_ring, ntime_gulp=NTIME_GULP,
        testfile=TESTFILE_NAME, coeffs=coeffs))

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
