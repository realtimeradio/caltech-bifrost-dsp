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
        self.ninput = ninput
        self.freqs = np.zeros(self.nchan, dtype=np.float32)
        
        # Setup the beamformer
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        ## Delays and gains
        self.cal_gains = np.zeros((nbeam, nchan, ninput), dtype=np.complex64) #: calibration gains
        self.gains_cpu = np.zeros((nbeam,nchan,ninput), dtype=np.complex64) #: CPU-side beamformer coeffs to be copied
        self.gains_gpu = BFArray(shape=(nbeam,nchan,ninput), dtype=np.complex64, space='cuda') #: GPU-side beamformer coeffs

        self.define_command_key('coeffs', type=dict, initial_val={})

        # Initialize beamforming library
        if ntime_sum is not None:
            self.log.warning("Running Beamform block with ntime_sum != None is experimental!")
            _bf.bfBeamformInitialize(self.gpu, self.ninput, self.nchan, self.ntime_gulp, self.nbeam, self.ntime_blocks)
        else:
            _bf.bfBeamformInitialize(self.gpu, self.ninput, self.nchan, self.ntime_gulp, self.nbeam, 0)

    #def _compute_weights(self, sfreq, nchan, chan_bw):
    #    """
    #    Regenerate complex gains from
    #    sfreq: Center freq of first channel in Hz
    #    nchan: Number of frequencies
    #    chan_bw: Channel bandwidth in Hz
    #    """
    #    cpu_affinity.set_core(self.core)
    #    self.acquire_control_lock()
    #    freqs = np.arange(sfreq, sfreq+nchan*chan_bw, chan_bw)
    #    phases = 2*np.pi*np.exp(1j*(freqs[:,None,None] * self.delays*1e-9)) #freq x beam x antpol
    #    self.new_cgains[...] = (phases * self.gains).transpose([1,0,2])
    #    self.release_control_lock()

    def _etcd_callback(self, watchresponse):
        """
        A callback executed whenever this block's command key is modified.

        This callback JSON decodes the key contents, and passes the
        resulting dictionary to ``_process_commands``.
        The ``last_cmd_response`` status value is set to the return value of
        ``_process_commands`` to indicate any error conditions

        :param watchresponse: A WatchResponse object used by the etcd
            `add_watch_prefix_callback` as the calling argument.
        :type watchresponse: WatchResponse
        """
        # We expect update commands to come frequently, and commands
        # all share the same command key. So, each command must be enacted
        # immediately (i.e, update_command_vals() should be run) in order
        # for newer commands not to overwrite older ones
        cpu_affinity.set_core(self.core)
        self.acquire_control_lock()
        for event in watchresponse.events:
            v = json.loads(event.value)
            self.update_stats({'last_cmd_response':self._process_commands(v)})
            self.update_command_vals()
        self.release_control_lock()

    def update_command_vals(self):
        """
        Copy command entries from the ``_pending_command_vals``
        dictionary to the ``command_vals`` dictionary, to be used
        by the block's runtime processing.
        Set the ``update_pending`` flag to False, to indicate that 
        there are no longer waiting commands. Set the status key
        ``last_cmd_proc_time`` to ``time.time()`` to record the
        time at which this method was called.
        """
        cpu_affinity.set_core(self.core)
        self.command_vals.update(self._pending_command_vals)
        for k, v in self._pending_command_vals.items():
           try:
               if v['type'] == 'gains':
                   i = v['input_id']
                   b = v['beam_id']
                   self.log.debug("BEAMFORM >> Updating calibration gains for beam %d, input %d" % (b,i))
                   data = np.array(v['data'])
                   self.cal_gains[b, :, i] = data[0::2] + 1j*data[1::2]
               if v['type'] == 'delays':
                   b = v['beam_id']
                   self.log.debug("BEAMFORM >> Updating delays for beam %d" % (b))
                   data = np.array(v['data'])
                   phases = 2*np.pi*np.exp(1j*self.freqs[:, None]*data*1e-9) # freq x pol
                   self.gains_cpu[b] = phases * self.cal_gains[b]
           except KeyError:
               self.log.error("BEAMFORM >> Failed to parse command")
        self.update_stats(self.command_vals)

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        igulp_size = self.ntime_gulp   * self.nchan * self.ninput    # 4+4
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
                self.freqs = np.arange(sfreq, sfreq+nchan*chan_bw, chan_bw)
                
                oshape = (self.ntime_gulp,nchan,self.nbeam*2)
                
                freqs = np.arange(sfreq, sfreq+nchan*chan_bw, chan_bw)

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
                            self.acquire_control_lock()
                            self.update_pending = False
                            self.stats['update_pending'] = False
                            self.stats['last_cmd_proc_time'] = time.time()
                            self.log.debug("BEAMFORM >> Copy coefficients to GPU")
                            self.gains_gpu[...] = self.gains_cpu
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
                            
                            _bf.bfBeamformRun(idata.as_BFarray(), odata.as_BFarray(), self.gains_gpu.as_BFarray())
                            BFSync()
                            
                        ## Update the base time tag
                        base_time_tag += self.ntime_gulp
                        
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': 8*igulp_size / process_time / 1e9})
