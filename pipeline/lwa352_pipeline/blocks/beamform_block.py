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
import ujson as json
import numpy as np

from .block_base import Block

class Beamform(Block):
    # Note: Input data are: [time,chan,ant,pol,cpx,8bit]
    """
    **Functionality**

    This block reads data from a GPU-side bifrost ring buffer and feeds
    it to to a beamformer, generating either voltage or (integrated)
    power beams.

    **New Sequence Condition**

    This block starts a new sequence each time the upstream sequence changes.

    **Input Header Requirements**

    This block requires that the following header fields
    be provided by the upstream data source:

    .. table::
        :widths: 25 10 10 55

        +-------------+--------+-------+------------------------------------------------+
        | Field       | Format | Units | Description                                    |
        +=============+========+=======+================================================+
        | ``seq0``    | int    |       | Spectra number for the first sample in the     |
        |             |        |       | input sequence                                 |
        +-------------+--------+-------+------------------------------------------------+
        | ``nchan``   | int    |       | Number of channels in the sequence             |
        +-------------+--------+-------+------------------------------------------------+
        | ``sfreq``   | double | Hz    | Center frequency of first channel in the       |
        |             |        |       | sequence                                       |
        +-------------+--------+-------+------------------------------------------------+
        | ``bw_hz``   | int    | Hz    | Bandwidth of the sequence                      |
        +-------------+--------+-------+------------------------------------------------+
        | ``nstand``  | int    |       | Number of stands (antennas) in the sequence    |
        +-------------+--------+-------+------------------------------------------------+
        | ``npol``    | int    |       | Number of polarizations per stand in the       |
        |             |        |       | sequence                                       |
        +-------------+--------+-------+------------------------------------------------+

    **Output Headers**

    This block passes headers from the upstream block with
    the following modifications:

    .. table::
        :widths: 25 10 10 55

        +------------------+----------------+---------+-------------------------------+
        | Field            | Format         | Units   | Description                   |
        +==================+================+=========+===============================+
        | ``nstand``       | int            |         | Number of beams in the        |
        |                  |                |         | sequence                      |
        +------------------+----------------+---------+-------------------------------+
        | ``nbeam``        | int            |         | Number of beams in the        |
        |                  |                |         | sequence                      |
        +------------------+----------------+---------+-------------------------------+
        | ``nbit``         | int            |         | Number of bits per output     |
        |                  |                |         | sample. This block sets this  |
        |                  |                |         | value to 32                   |
        +------------------+----------------+---------+-------------------------------+
        | ``npol``         | int            |         | Number of polarizations per   |
        |                  |                |         | beam. This block sets this    |
        |                  |                |         | value to 1.                   |
        +------------------+----------------+---------+-------------------------------+
        | ``complex``      | Bool           |         | This block sets this entry to |
        |                  |                |         | ``True``, indicating that the |
        |                  |                |         | data out of this block are    |
        |                  |                |         | complex, with real and        |
        |                  |                |         | imaginary parts each of width |
        |                  |                |         | ``nbit``                      |
        +------------------+----------------+---------+-------------------------------+

    **Data Buffers**

    *Input Data Buffer*: A GPU-side bifrost ring buffer of 4+4 bit
    complex data in order: ``time x channel x input``.

    Typically, the ``input`` axis is composed of ``stand x polarization``,
    but this block does not assume this is the case.

    Each gulp of the input buffer reads ``ntime_gulp`` samples, I.e
    ``ntime_gulp x nchan x ninput`` bytes.

    *Output Data Buffer*: A GPU-side bifrost ring buffer of 32+32 bit complex
    floating-point  data containing beamformed data. With ``ntime_sum=None``, this is
    complex beamformer data with dimensionality
    ``time x channel x beams x complexity``. This output buffer is written
    in blocks of ``ntime_gulp`` samples, I.e. ``ntime_gulp x nchan x nbeam x 8`` bytes.

    With ``ntime_sum != None``, this block will generate dynamic power spectra
    rather than voltages. This mode is experimental. See ``bifrost/beamform.h``
    for details.


    **Instantiation**

    :param log: Logging object to which runtime messages should be
        emitted.
    :type log: logging.Logger

    :param iring: bifrost input data ring. This should be on the GPU.
    :type iring: bifrost.ring.Ring

    :param oring: bifrost output data ring. This should be on the GPU.
    :type oring: bifrost.ring.Ring

    :param guarantee: If True, read data from the input ring in blocking "guaranteed"
        mode, applying backpressure if necessary to ensure no data are missed by this block.
    :type guarantee: Bool

    :param core: CPU core to which this block should be bound. A value of -1 indicates no binding.
    :type core: int

    :param gpu: GPU device which this block should target. A value of -1 indicates no binding
    :type gpu: int

    :param ntime_gulp: Number of time samples to copy with each gulp.
    :type ntime_gulp: int

    :param nchan: Number of frequency channels per time sample. This should match
        the sequence header ``nchan`` value, else an AssertionError is raised.
    :type nchan: int

    :param ninput: Number of inputs per time sample. This should match
        the sequence headers ``nstand x npol``, else an AssertionError is raised.
    :type nstand: int

    :param ntime_sum: Set to ``None`` to generate voltage beams. Set to an integer
        value >=0 to generate accumulated dynamic spectra. Values other than
        None are *experimental*.
    :type ntime_sum: int

    :parameter etcd_client: Etcd client object used to facilitate control of this block.
        If ``None``, do not use runtime control.
    :type etcd_client: etcd3.client.Etcd3Client

    **Runtime Control and Monitoring**

    This block accepts the following command fields:

    .. table::
        :widths: 25 10 10 55

        +------------------+-----------------+--------+-------------------------------------------------+
        | Field            | Format          | Units  | Description                                     |
        +==================+=================+========+=================================================+
        | ``delays``       | 2D list of      | ns     | An ``nbeam x ninput`` element list of geometric |
        |                  | float           |        | delays, in nanoseconds.                         |
        +------------------+-----------------+--------+-------------------------------------------------+
        | ``gains``        | 2D list of      |        | A two dimensional list of calibration gains     |
        |                  | complex32)      |        | with shape ``nchan x ninput``                   |
        +------------------+-----------------+--------+-------------------------------------------------+
        | ``load_sample``  | int             | sample | **NOT YET IMPLEMENTED** Sample number on which  |
        |                  |                 |        | the supplied delays should be loaded. If this   |
        |                  |                 |        | field is absent, new delays will be loaded as   |
        |                  |                 |        | soon as possible.                               |
        +------------------+-----------------+--------+-------------------------------------------------+

    """

    def __init__(self, log, iring, oring, nchan=256, nbeam=1, ninput=352*2, ntime_gulp=2500, ntime_sum=None, guarantee=True, core=-1, gpu=-1, etcd_client=None):

        super(Beamform, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)

        self.ntime_gulp = ntime_gulp
        self.gpu = gpu
        self.ntime_sum = ntime_sum
        if ntime_sum is not None:
            assert ntime_gulp % ntime_sum == 0
            self.ntime_blocks = ntime_gulp // ntime_sum
        else:
            self.ntime_blocks = ntime_gulp
        
        self.nchan = nchan
        self.nbeam = nbeam
        self.ninput = self.ninput
        
        # Setup the beamformer
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        ## Delays and gains
        self.delays = np.zeros((nbeam, ninput), dtype=np.float64)
        self.gains = np.zeros((nbeam, ninput), dtype=np.float64)
        self.new_cgains = np.zeros((nbeam,nchan,ninput), dtype=np.complex64)
        self.cgains = BFArray(shape=(nbeam,nchan,ninput), dtype=np.complex64, space='cuda')
        self.update_pending = True

        # Initialize beamforming library
        if ntime_sum is not None:
            self.log.warning("Running Beamform block with ntime_sum != None is experimental!")
            _bf.bfBeamformInitialize(self.gpu, self.ninput, self.nchan, self.ntime_gulp, self.nbeam, self.ntime_blocks)
        else:
            _bf.bfBeamformInitialize(self.gpu, self.ninput, self.nchan, self.ntime_gulp, self.nbeam, 0)


    def _etcd_callback(self, watchresponse):
        v = json.loads(watchresponse.events[0].value)
        #self.acquire_control_lock()
        if 'delays' in v and isinstance(v['delays'], list):
            self.delays[...] = v['delays']
        if 'gains' in v and isinstance(v['gains'], list):
            self.gains[...] = v['gains']
        self.update_pending = True
        #self.release_control_lock()

    def _compute_weights(self, sfreq, nchan, chan_bw):
        """
        Regenerate complex gains from
        sfreq: Center freq of first channel in Hz
        nchan: Number of frequencies
        chan_bw: Channel bandwidth in Hz
        """
        cpu_affinity.set_core(self.core)
        self.acquire_control_lock()
        freqs = np.arange(sfreq, sfreq+nchan*chan_bw, chan_bw)
        phases = 2*np.pi*np.exp(1j*(freqs[:,None,None] * self.delays*1e-9)) #freq x beam x antpol
        self.new_cgains[...] = (phases * self.gains).transpose([1,0,2])
        self.release_control_lock()

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        igulp_size = self.ntime_gulp   * self.nchan * self.ninputs       # 4+4
        ogulp_size = self.ntime_blocks * self.nchan * self.nbeam * 8 # complex 64

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                # recalculate beamforming coefficients on each new sequence (freqs could have changed)
                self.update_pending = True
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                sfreq  = ihdr['sfreq']
                bw     = ihdr['bw_hz']
                chan_bw  = bw / nchan

                assert nchan == self.nchan
                assert self.ninput == nstand * npol
                
                oshape = (self.ntime_gulp,nchan,self.nbeam*2)
                
                freqs = np.arange(sfreq, sfreq+nchan*chan_bw, chan_bw)

                ticksPerTime = int(FS) / int(CHAN_BW)
                base_time_tag = iseq.time_tag
                
                ohdr = ihdr.copy()
                ohdr['nstand'] = self.nbeam
                ohdr['nbit'] = 32
                ohdr['npol'] = 1 # The beamformer inherently produces single-pol beams
                ohdr['complex'] = True
                ohdr['nbeam'] = self.nbeam
                ohdr_str = json.dumps(ohdr)
                
                self.oring.resize(ogulp_size)
                
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    for ispan in iseq.read(igulp_size):
                        if ispan.size < igulp_size:
                            continue # Ignore final gulp
                        if self.update_pending:
                            self.log.info("BEAMFORM >> Updating coefficients")
                            self._compute_weights(ohdr['sfreq'], ohdr['nchan'], ohdr['bw_hz']/ohdr['nchan'])
                            self.acquire_control_lock()
                            # Copy data to GPU
                            self.cgains[...] = self.new_cgains
                            #self.cgains[0] = self.new_cgains[0]
                            self.update_pending = False
                            self.release_control_lock()
                        
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        with oseq.reserve(ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            ## Setup and load
                            idata = ispan.data_view('i8')
                            odata = ospan.data_view(np.float32)
                            
                            _bf.bfBeamformRun(idata.as_BFarray(), odata.as_BFarray(), self.cgains.as_BFarray())
                            BFSync()
                            
                        ## Update the base time tag
                        base_time_tag += self.ntime_gulp*ticksPerTime
                        
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': 8*igulp_size / process_time / 1e9})
