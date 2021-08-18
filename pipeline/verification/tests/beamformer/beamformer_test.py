#! /usr/bin/env python

import sys
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

NSTAND = 352
NPOL = 2
NTIME_GULP = 480
NCHAN = 96
GPUDEV = 0
ETCD_HOST = "etcdv3service.sas.pvt"
TESTFILE_NAME = '/tmp/temp_testfile.dat'
NBEAM = 16

CORES = list(range(12))

class BeamPrinter(Block):
    def __init__(self, log, iring, ntime_gulp=2500,
                 guarantee=True, core=-1):

        super(BeamPrinter, self).__init__(log, iring, None, guarantee, core, etcd_client=None)
        cpu_affinity.set_core(self.core)
        self.ntime_gulp = ntime_gulp

    def main(self):
        cpu_affinity.set_core(self.core)

        tick = time.time()
        for iseq in self.iring.read(guarantee=self.guarantee):
            gulpi = 0
            ihdr = json.loads(iseq.header.tostring())
            nstand = ihdr['nstand']
            npol = ihdr['npol']
            nbeam = ihdr['nbeam']
            nchan = ihdr['nchan']
            #print(ihdr)
            igulp_size = self.ntime_gulp * nchan * nbeam * 2 * 4 # complex float32
            for ispan in iseq.read(igulp_size):
                idata = ispan.data_view('cf32').reshape([nchan, nbeam, self.ntime_gulp])
                if time.time() - tick > 1:
                    print('Gulp: %d' % gulpi)
                    print('Time (beam0, chan0):', idata[0, 0, 0:10])
                    print('Chan (time 0, beam 0):', idata[:,0,0])
                    print('Beam (time 0, chan0):', idata[0,:,0])
                    tick = time.time()
                gulpi += 1

def main():
    test_data = np.ones([NTIME_GULP, NCHAN, NSTAND, NPOL], dtype=np.uint8)
    #for s in range(NSTAND):
    #    test_data[:, :, s, :] = (NSTAND % 8) << 4
    with open(TESTFILE_NAME, 'wb') as fh:
        print("Writing testfile: %s" % TESTFILE_NAME)
        fh.write(test_data.tobytes())
    
    etcd_client = etcd.client(ETCD_HOST)

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

    ops.append(Copy(log, iring=bf_output_ring, oring=output_ring, ntime_gulp=NTIME_GULP,
                      nbyte_per_time=2*NBEAM*NCHAN*2*4,
                      core=CORES.pop(0), guarantee=True, gpu=GPUDEV))

    ops.append(BeamPrinter(log, iring=output_ring, ntime_gulp=NTIME_GULP))

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
