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

from .block_base import Block

class BeamformSumBeams(Block):
    """
    **Functionality**

    This block reads beamformed voltage data from a GPU-side ring buffer and
    generates integrated power spectra.

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
        | ``nstand``  | int    |       | Number of stands (antennas) in the sequence    |
        +-------------+--------+-------+------------------------------------------------+
        | ``nbeam``   | int    |       | Number of single polarization beams in the     |
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
        | ``nbeam``        | int            |         | Number of dual polatization   |
        |                  |                |         | beams in the sequence. Equal  |
        |                  |                |         | to half the input header      |
        |                  |                |         | ``nbeam``                     |
        +------------------+----------------+---------+-------------------------------+
        | ``nbit``         | int            |         | Number of bits per output     |
        |                  |                |         | sample. This block sets this  |
        |                  |                |         | value to 32                   |
        +------------------+----------------+---------+-------------------------------+
        | ``npol``         | int            |         | Number of polarizations per   |
        |                  |                |         | beam. This block sets this    |
        |                  |                |         | value to 2.                   |
        +------------------+----------------+---------+-------------------------------+
        | ``complex``      | Bool           |         | This block sets this entry to |
        |                  |                |         | ``True``, indicating that the |
        |                  |                |         | data out of this block are    |
        |                  |                |         | complex, with real and        |
        |                  |                |         | imaginary parts each of width |
        |                  |                |         | ``nbit``                      |
        +------------------+----------------+---------+-------------------------------+
        | ``acc_len``      | int            |         | Number of spectra integrated  |
        |                  |                |         | into each output sample by    |
        |                  |                |         | this block                    |
        +------------------+----------------+---------+-------------------------------+

    **Data Buffers**

    *Input Data Buffer*: A GPU-side bifrost ring buffer of 32+32 bit complex
    floating-point data containing beamformed voltages.
    The input buffer has dimensionality (slowest varying to fastest varying)
    ``time x channel x beams x complexity``. 
    The number of beams should be even.

    Each gulp of the input buffer reads ``ntime_gulp`` samples, I.e
    ``ntime_gulp x nchan x nbeam x 8`` bytes.

    This block considers beam indices ``0, 2, 4, ..., nbeam-2`` to be ``X`` polarized,
    and beam indices ``1, 3, 5, ..., nbeam-1`` to be ``Y`` polarized. As such, ``nbeam/2``
    output beams are generated by this block, with 4 polarization products each.

    *Output Data Buffer*: A CPU- or GPU-side bifrost ring buffer of 32 bit,
    floating-point, integrated, beam powers.
    Data has dimensionality ``time x channel x beams x beam-element``.

    ``channel`` runs from 0 to ``nchan``.
    
    ``beam`` runs from 0 to the output ``nbeam-1`` (equivalent to the input ``nbeam/2 - 1``).
    
    ``beam-element`` runs from 0 to 3 with the following mapping:

      - index 0: The accumulated power of a beam's ``X`` polarization
      - index 1: The accumulated power of a beam's ``Y`` polarization
      - index 2: The accumulated real part of a beam's ``X x conj(Y)`` cross-power
      - index 3: The accumulated imaginary part of a beam's ``X x conj(Y)`` cross-power

    This block sums over ``ntime_sum`` input time samples, thus writing ``ntime_gulp / ntime_sum``
    output samples for every ``ntime_gulp`` samples read.
    
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

    :param ntime_sum: The number of time sample which this block should integrate.
        ``ntime_gulp`` should be an integer multiple of ``ntime_sum``, else an
        AssertionError is raised.
    :type ntime_sum: int

    **Runtime Control and Monitoring**

    This block has no runtime control keys. It is completely configured at instantiation time.

    """

    def __init__(self, log, iring, oring, nchan=256,
                 ntime_gulp=2500, ntime_sum=24, guarantee=True, core=-1, gpu=-1,
                 etcd_client=None):

        super(BeamformSumBeams, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)

        self.ntime_gulp = ntime_gulp
        self.gpu = gpu
        self.ntime_sum = ntime_sum
        assert ntime_gulp % ntime_sum == 0
        self.ntime_blocks = ntime_gulp // ntime_sum
        
        self.nchan = nchan

        #self.ogulp_size = self.ntime_blocks * self.nchan * 4 * 4 # 4 x float32

        ## The output gulp size can be quite small if we base it on the input gulp size
        ## force the numper of times in the output span to match the input, which
        ## is likely to be more reasonable
        #self.oring.resize(self.ntime_gulp * self.nchan * 4 * 4)

        # Setup the beamformer
        if self.gpu != -1:
            BFSetGPU(self.gpu)

        # Don't Initialize beamforming library -- this should have been done by the beamformer already
        #_bf.bfBeamformInitialize(self.gpu, self.ninputs, self.nchan, self.ntime_gulp, self.nbeam_max, self.ntime_blocks)

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                nchan  = ihdr['nchan']
                nbeam  = ihdr['nbeam']
                assert nchan == self.nchan
                
                ticksPerTime = int(FS) / int(CHAN_BW)
                base_time_tag = iseq.time_tag
                
                ohdr = ihdr.copy()
                ohdr['nbeam'] = nbeam // 2 #Go from single pol beams to dual pol
                ohdr['nbit'] = 32
                ohdr['complex'] = True
                ohdr['acc_len'] = self.ntime_sum
                ohdr['npol'] = 2 # Forces dual pol output by combining pairs of beams as if X/Y
                ohdr_str = json.dumps(ohdr)

                # Block output
                self.bf_output = BFArray(shape=(self.ntime_blocks, ohdr['nbeam'], nchan, 4), dtype=np.float32, space='cuda')
                igulp_size = self.ntime_gulp * nchan * ihdr['nbeam'] * 2 * ihdr['nbit'] // 8
                ogulp_size = self.ntime_blocks * nchan * ohdr['nbeam'] * 4 * 4 # 4 x float32 per sample
                # The output gulp size can be quite small if we base it on the input gulp size
                # force the numper of times in the output span to match the input, which
                # is likely to be more reasonable
                self.oring.resize(ogulp_size // self.ntime_blocks * self.ntime_gulp * 4)
                
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
                            odata = ospan.data_view(np.float32).reshape(self.bf_output.shape)
                            _bf.bfBeamformIntegrate(idata.as_BFarray(), self.bf_output.as_BFarray(), self.ntime_sum)
                            odata[...] = self.bf_output
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
