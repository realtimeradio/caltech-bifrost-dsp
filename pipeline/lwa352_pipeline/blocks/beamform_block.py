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

from .block_base import Block, COMMAND_OK, COMMAND_INVALID

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
    ``channel x beams x time x complexity``. This output buffer is written
    in blocks of ``ntime_gulp`` samples, I.e. ``nchan x nbeam x ntime_gulp x 8`` bytes.

    With ``ntime_sum != None``, this block will generate dynamic power spectra
    rather than voltages. This mode is experimental (aka probably doesn't work).
    See ``bifrost/beamform.h`` for details.


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
        | ``coeffs``       | dict            |        | A dictionary with coefficient data              |
        +------------------+-----------------+--------+-------------------------------------------------+

    The ``coeffs`` command dictionary contains the following fields.

        +------------------+-----------------+--------+-------------------------------------------------+
        | Field            | Format          | Units  | Description                                     |
        +==================+=================+========+=================================================+
        | ``type``         | string          |        | A string describing the command type. May be    |
        |                  |                 |        | 'calgains' for calibration gains, or            |
        |                  |                 |        | 'beamcoeffs' for beam coefficients.             |
        +------------------+-----------------+--------+-------------------------------------------------+
        | ``input_id``     | int             |        | Input index of the signal to which these        |
        |                  |                 |        | calibration gains should be applied if          |
        |                  |                 |        | ``type=='calgains'``. Unused if                 |
        |                  |                 |        | ``type=='beamcoeffs'``.                         |
        +------------------+-----------------+--------+-------------------------------------------------+
        | ``beam_id``      | int             |        | Beam index which these beamforming coefficients |
        |                  |                 |        | should be applied.                              |
        +------------------+-----------------+--------+-------------------------------------------------+
        | ``data``         |                 |        | If ``type==calgains`` a list of floats of       |
        |                  |                 |        | length ``2*nchan``. Entry ``2i`` of this list   |
        |                  |                 |        | is the real part of the calibration gain for    |
        |                  |                 |        | frequency channel ``i``. Entry ``2i+1`` is the  |
        |                  |                 |        | imaginary part of the calibration gain for      |
        |                  |                 |        | frequency channel ``i``. If                     |
        |                  |                 |        | ``type==beamcoeffs``, ``data`` is a dictionary  |
        |                  |                 |        | with up to 3 keys: ``delays`` is an             |
        |                  |                 |        | ``nsignal``-length list of floating point       |
        |                  |                 |        | delays, in units of nanoseconds. Entry ``i`` of |
        |                  |                 |        | this list is the delay which should be applied  |
        |                  |                 |        | to signal ``i``. to signal ``i``. ``amps`` is   |
        |                  |                 |        | an ``nsignal``-length list of floating point    |
        |                  |                 |        | amplitudes, with entry ``i`` being the          |
        |                  |                 |        | amplitude to apply to signal ``i``.             |
        |                  |                 |        | ``load_sample`` is the integer sample number at |
        |                  |                 |        | which the provided amplitudes and delays should |
        |                  |                 |        | be loaded. If not provided, coefficients are    |
        |                  |                 |        | loaded immediately.                             |
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
        self.cal_gains = np.ones((nchan, nbeam, ninput), dtype=np.complex64) #: calibration gains
        # Three buffers for complex gains.
        #   `gains_cpu_new` is the latest set of coefficients. As soon as new coefficients arrive
        #   they are loaded here.
        #   `gains_cpu is` the set of coefficient array waiting to be copied to the GPU. coefficients
        #   from gains_cpu_new are copied to gains_cpu when a user-supplied trigger time is reached
        #   `gains_gpu` is the coefficient set in GPU device memory. Whenever gains_cpu_new changes,
        #   these coefficients are copied. Copies go through an intermediate CPU buffer so that all
        #   transfers to GPU memory are contiguous.
        self.gains_cpu_new = np.zeros((nchan, nbeam, ninput), dtype=np.complex64) #: CPU-side beamformer coeffs waiting to be activated
        self.gains_cpu = np.zeros((nchan, nbeam, ninput), dtype=np.complex64) #: CPU-side beamformer coeffs to be copied
        self.gains_gpu = BFArray(shape=(nchan, nbeam, ninput), dtype=np.complex64, space='cuda') #: GPU-side beamformer coeffs
        self.gains_load_sample = np.zeros(nbeam) #: sample time at which gains_cpu_new should be copied to gains_cpu (and on to gains_gpu)

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
            seq_id = v.get('id', None)
            if seq_id is None:
                self._send_command_response("0", False, "Missing ID field")
                continue
            cmd = v.get('cmd', None)
            if cmd != "update":
                self._send_command_response("0", False, "Invalid command")
                continue
            val = v.get("val", None)
            if not isinstance(val, dict):
                self._send_command_response(seq_id, False, "`val` field should be a dictionary")
                continue
            update_keys = val.get("kwargs", None)
            if not isinstance(update_keys, dict):
                self._send_command_response(seq_id, False, "`val[kwargs]` field should be a dictionary")
                continue
            try:
                proc_ok = self._process_commands(update_keys, set_pending_flag=False)
            except:
                proc_ok = COMMAND_INVALID
            self.update_stats({'last_cmd_response':proc_ok})
            self.update_command_vals()
            self._send_command_response(seq_id, proc_ok==COMMAND_OK, str(proc_ok))
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
               if v['type'] == 'calgains':
                   i = v['input_id']
                   b = v['beam_id']
                   self.log.debug("BEAMFORM >> Updating calibration gains for beam %d, input %d" % (b,i))
                   data = np.array(v['data'])
                   self.cal_gains[:, b, i] = data[0::2] + 1j*data[1::2] # freq x beam x input
                   self.update_stats({'cal_gains%d' % b: True})
               if v['type'] == 'beamcoeffs':
                   b = v['beam_id']
                   self.log.debug("BEAMFORM >> Updating delays for beam %d" % (b))
                   delays_ns = np.array(v['data']['delays'])
                   amps = np.array(v['data']['amps'])
                   phases = np.exp(1j*2*np.pi*self.freqs[:, None]*delays_ns*1e-9) # freq x input
                   self.gains_cpu_new[:, b, :] = amps * phases * self.cal_gains[:, b, :] # freq x beam x input
                   self.gains_load_sample[b] = v.get('load_sample', -1) # default to immediate load
                   # Only trigger update on beamcoeffs, not calibration only.
                   # This means loading [lots of] calibration data has less of an impact on the
                   # pipeline performance. Calibrations are only loaded after beamformer weights are applied.
                   self.update_pending = True # Only trigger update on beamcoeffs, not calibration only
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
        oshape = (self.ntime_gulp,self.nchan,self.nbeam*2)
        self.oring.resize(ogulp_size)

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                # recalculate beamforming coefficients on each new sequence (freqs could have changed)
                self.update_pending = True
                copy_pending = True
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                this_gulp_time = ihdr['seq0']
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                sfreq  = ihdr['sfreq']
                bw     = ihdr['bw_hz']
                chan_bw  = bw / nchan

                assert nchan == self.nchan
                assert self.ninput == nstand * npol
                self.freqs = np.arange(sfreq, sfreq+nchan*chan_bw, chan_bw)
                
                freqs = np.arange(sfreq, sfreq+nchan*chan_bw, chan_bw)

                
                ohdr = ihdr.copy()
                ohdr['nstand'] = self.nbeam
                ohdr['nbit'] = 32
                ohdr['npol'] = 1 # The beamformer inherently produces single-pol beams
                ohdr['complex'] = True
                ohdr['nbeam'] = self.nbeam
                ohdr_str = json.dumps(ohdr)
                
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    for ispan in iseq.read(igulp_size):
                        self.update_stats({'curr_sample': this_gulp_time})
                        if ispan.size < igulp_size:
                            continue # Ignore final gulp
                        if self.update_pending:
                            self.acquire_control_lock()
                            for b in range(self.nbeam):
                                if self.gains_load_sample[b] == 0:
                                    continue
                                if this_gulp_time >= self.gains_load_sample[b]:
                                    self.gains_cpu[:,b,:] = self.gains_cpu_new[:,b,:]
                                    self.gains_load_sample[b] = 0
                                    copy_pending = True
                            if self.gains_load_sample.sum() == 0:
                                self.update_pending = False
                            self.stats['update_pending'] = self.update_pending
                            self.stats['last_cmd_proc_time'] = time.time()
                            self.release_control_lock()
                        
                        if copy_pending:
                            self.log.debug("BEAMFORM >> Copy coefficients to GPU at time %d" % this_gulp_time)
                            self.gains_gpu[...] = self.gains_cpu
                            copy_pending = False

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
                        this_gulp_time += self.ntime_gulp
                        
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': 8*igulp_size / process_time / 1e9})
