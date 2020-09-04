import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap
from bifrost.ndarray import copy_array
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, stream_synchronize as BFSync
from bifrost.unpack import unpack

import time
import simplejson as json
import numpy as np
from collections import deque

from blocks.block_base import Block

FS=200.0e6 # sampling rate
CLOCK            = 204.8e6 #???
NCHAN            = 4096
FREQS            = np.around(np.fft.fftfreq(2*NCHAN, 1./CLOCK)[:NCHAN][:-1], 3)
CHAN_BW          = FREQS[1] - FREQS[0]

class BeamformSum(Block):
    # Note: Input data are: [time,chan,ant,pol,cpx,8bit]
    def __init__(self, log, iring, oring, tuning=0, nchan_max=256, nbeam_max=1, nstand=352, npol=2, ntime_gulp=2500, ntime_sum=24, guarantee=True, core=-1, gpu=-1, etcd_client=None):

        super(BeamformSum, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)

        self.tuning = tuning
        self.ntime_gulp = ntime_gulp
        self.gpu = gpu
        self.ntime_sum = ntime_sum
        assert ntime_gulp % ntime_sum == 0
        self.ntime_blocks = ntime_gulp // ntime_sum
        
        self.nchan_max = nchan_max
        self.nbeam_max = nbeam_max
        self.nstand = nstand
        self.npol = npol
        self.ninputs = nstand*npol

        # TODO self.configMessage = ISC.BAMConfigurationClient(addr=('adp',5832))
        self._pending = deque()
        
        # Setup the beamformer
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        ## Metadata
        nchan = self.nchan_max

        # Block output
        self.bf_output = BFArray(shape=(self.nbeam_max, self.ntime_blocks, self.nchan_max, 4), dtype=np.float32, space='cuda')

        # Initialize beamforming library
        _bf.bfBeamformInitialize(self.gpu, self.ninputs, self.nchan_max, self.ntime_gulp, self.nbeam_max, self.ntime_blocks)

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        igulp_size = self.ntime_gulp   * self.nchan_max * self.nbeam_max * 8 #complex 64
        ogulp_size = self.ntime_blocks * self.nchan_max * self.nbeam_max * 8 #complex 64

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                
                #ishape = (self.ntime_gulp,nchan,self.nbeam_max*2)
                #oshape = (self.ntime_blocks,nchan,self.nbeam_max*2)
                
                ticksPerTime = int(FS) / int(CHAN_BW)
                base_time_tag = iseq.time_tag
                
                ohdr = ihdr.copy()
                ohdr['nstand'] = self.nbeam_max
                ohdr['nbit'] = 32
                ohdr['complex'] = True
                ohdr_str = json.dumps(ohdr)
                
                self.oring.resize(ogulp_size)
                
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    for ispan in iseq.read(igulp_size):
                        if ispan.size < igulp_size:
                            continue # Ignore final gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        with oseq.reserve(ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            ## Setup and load
                            idata = ispan.data_view(np.float32)
                            odata = ospan.data_view(np.float32)
                            
                            _bf.bfBeamformIntegrate(idata.as_BFarray(), self.bf_output.as_BFarray())
                            odata = self.bf_output.data
                            BFSync()
                            
                        ## Update the base time tag
                        base_time_tag += self.ntime_gulp*ticksPerTime
                        
                        ## Check for an update to the configuration
                        
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': 8*igulp_size / process_time / 1e9})
