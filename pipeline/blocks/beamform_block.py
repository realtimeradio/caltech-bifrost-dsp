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

class Beamform(Block):
    # Note: Input data are: [time,chan,ant,pol,cpx,8bit]
    def __init__(self, log, iring, oring, tuning=0, nchan_max=256, nbeam_max=1, nstand=352, npol=2, ntime_gulp=2500, ntime_sum=24, guarantee=True, core=-1, gpu=-1, etcd_client=None):

        super(Beamform, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)

        self.tuning = tuning
        self.ntime_gulp = ntime_gulp
        self.gpu = gpu
        self.ntime_sum = ntime_sum
        if ntime_sum is not None:
            assert ntime_gulp % ntime_sum == 0
            self.ntime_blocks = ntime_gulp // ntime_sum
        else:
            self.ntime_blocks = ntime_gulp
        
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
        ## Delays and gains
        self.delays = np.zeros((self.nbeam_max*2,nstand*npol), dtype=np.float64)
        self.gains = np.zeros((self.nbeam_max*2,nstand*npol), dtype=np.float64)
        self.cgains = BFArray(shape=(self.nbeam_max,nchan,nstand*npol), dtype=np.complex64, space='cuda')
        # Block output
        self.bf_output = BFArray(shape=(self.nbeam_max, self.ntime_blocks, self.nchan_max, 4), dtype=np.float32, space='cuda')

        # Initialize beamforming library
        if ntime_sum is not None:
            _bf.bfBeamformInitialize(self.gpu, self.ninputs, self.nchan_max, self.ntime_gulp, self.nbeam_max, self.ntime_blocks)
        else:
            _bf.bfBeamformInitialize(self.gpu, self.ninputs, self.nchan_max, self.ntime_gulp, self.nbeam_max, 0)

    def configMessage(self):
        return None
        
    #@ISC.logException
    def updateConfig(self, config, hdr, time_tag, forceUpdate=False):
        return True
        if self.gpu != -1:
            BFSetGPU(self.gpu)
            
        # Get the current pipeline time to figure out if we need to shelve a command or not
        pipeline_time = time_tag / FS
        
        # Can we act on this configuration change now?
        if config:
            ## Pull out the tuning (something unique to DRX/BAM/COR)
            beam, tuning = config[0], config[3]
            if beam > self.nbeam_max or tuning != self.tuning:
                return False
                
            ## Set the configuration time - BAM commands are for the specified slot in the next second
            slot = config[4] / 100.0
            config_time = int(time.time()) + 1 + slot
            
            ## Is this command from the future?
            if pipeline_time < config_time:
                ### Looks like it, save it for later
                self._pending.append( (config_time, config) )
                config = None
                
                ### Is there something pending?
                try:
                    stored_time, stored_config = self._pending[0]
                    if pipeline_time >= stored_time:
                        config_time, config = self._pending.popleft()
                except IndexError:
                    pass
            else:
                ### Nope, this is something we can use now
                pass
                
        else:
            ## Is there something pending?
            try:
                stored_time, stored_config = self._pending[0]
                if pipeline_time >= stored_time:
                    config_time, config = self._pending.popleft()
            except IndexError:
                #print "No pending configuation at %.1f" % pipeline_time
                pass
                
        if config:
            self.log.info("Beamformer: New configuration received for beam %i (delta = %.1f subslots)", config[0], (pipeline_time-config_time)*100.0)
            beam, delays, gains, tuning, slot = config
            if tuning != self.tuning:
                self.log.info("Beamformer: Not for this tuning, skipping")
                return False
                
            # Byteswap to get into little endian
            delays = delays.byteswap().newbyteorder()
            gains = gains.byteswap().newbyteorder()
            
            # Unpack and re-shape the delays (to seconds) and gains (to floating-point)
            delays = (((delays>>4)&0xFFF) + (delays&0xF)/16.0) / FS
            gains = gains/32767.0
            gains.shape = (gains.size/2, 2)
            
            # Update the internal delay and gain cache so that we can use these later
            self.delays[2*(beam-1)+0,:] = delays
            self.delays[2*(beam-1)+1,:] = delays
            self.gains[2*(beam-1)+0,:] = gains[:,0]
            self.gains[2*(beam-1)+1,:] = gains[:,1]
            
            # Compute the complex gains needed for the beamformer
            freqs = CHAN_BW * (hdr['chan0'] + np.arange(hdr['nchan']))
            freqs.shape = (freqs.size, 1)
            self.cgains[2*(beam-1)+0,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*(beam-1)+0,:]) * \
                                             self.gains[2*(beam-1)+0,:]).astype(np.complex64)
            self.cgains[2*(beam-1)+1,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*(beam-1)+1,:]) * \
                                             self.gains[2*(beam-1)+1,:]).astype(np.complex64)
            BFSync()
            self.log.info('  Complex gains set - beam %i' % beam)
            
            return True
            
        elif forceUpdate:
            self.log.info("Beamformer: New sequence configuration received")
            
            # Compute the complex gains needed for the beamformer
            freqs = CHAN_BW * (hdr['chan0'] + np.arange(hdr['nchan']))
            freqs.shape = (freqs.size, 1)
            for beam in xrange(1, self.nbeam_max+1):
                self.cgains[2*(beam-1)+0,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*(beam-1)+0,:]) \
                                                 * self.gains[2*(beam-1)+0,:]).astype(np.complex64)
                self.cgains[2*(beam-1)+1,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*(beam-1)+1,:]) \
                                                 * self.gains[2*(beam-1)+1,:]).astype(np.complex64)
                BFSync()
                self.log.info('  Complex gains set - beam %i' % beam)
                
            return True
            
        else:
            return False
        
    #@ISC.logException
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        igulp_size = self.ntime_gulp * self.nchan_max * self.ninputs         # 4+4
        ogulp_size = self.ntime_blocks * self.nchan_max * self.nbeam_max * 8 #complex 64

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                
                status = self.updateConfig( self.configMessage(), ihdr, iseq.time_tag, forceUpdate=True )
                
                ishape = (self.ntime_gulp,nchan,nstand*npol)
                oshape = (self.ntime_gulp,nchan,self.nbeam_max*2)
                
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
                            idata = ispan.data_view('i8')
                            odata = ospan.data_view(np.float32)#.reshape(oshape)
                            
                            #_bf.bfBeamformRun(idata.as_BFarray(), self.bf_output.as_BFarray(), self.cgains.as_BFarray())
                            _bf.bfBeamformRun(idata.as_BFarray(), odata.as_BFarray(), self.cgains.as_BFarray())
                            #odata = self.bf_output.data
                            BFSync()
                            
                        ## Update the base time tag
                        base_time_tag += self.ntime_gulp*ticksPerTime
                        
                        ## Check for an update to the configuration
                        self.updateConfig( self.configMessage(), ihdr, base_time_tag, forceUpdate=False )
                        
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': 8*igulp_size / process_time / 1e9})
