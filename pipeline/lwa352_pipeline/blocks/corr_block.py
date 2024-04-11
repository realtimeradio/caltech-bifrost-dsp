import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.device import set_device as BFSetGPU

import time
import ujson as json
import numpy as np

from .block_base import Block

## Computes the triangular index of an (i,j) pair as shown here...
## NB: Output is valid only if i >= j.
##
##      i=0  1  2  3  4..
##     +---------------
## j=0 | 00 01 03 06 10
##   1 |    02 04 07 11
##   2 |       05 08 12
##   3 |          09 13
##   4 |             14
##   :
def tri_index(i, j):
    return (i * (i+1))//2 + j;

## Returns index into the GPU's register tile ordered output buffer for the
## real component of the cross product of inputs in0 and in1.  Note that in0
## and in1 are input indexes (i.e. 0 based) and often represent antenna and
## polarization by passing (2*ant_idx+pol_idx) as the input number (NB: ant_idx
## and pol_idx are also 0 based).  Return value is valid if in1 >= in0.  The
## corresponding imaginary component is located xgpu_info.matLength words after
## the real component.
def regtile_index(in0, in1, nstand):
    nstation = nstand
    a0 = in0 >> 1;
    a1 = in1 >> 1;
    p0 = in0 & 1;
    p1 = in1 & 1;
    num_words_per_cell = 4;
  
    # Index within a quadrant
    quadrant_index = tri_index(a1//2, a0//2);
    # Quadrant for this input pair
    quadrant = 2*(a0&1) + (a1&1);
    # Size of quadrant
    quadrant_size = (nstation//2 + 1) * nstation//4;
    # Index of cell (in units of cells)
    cell_index = quadrant*quadrant_size + quadrant_index;
    #printf("%s: in0=%d, in1=%d, a0=%d, a1=%d, cell_index=%d\n", __FUNCTION__, in0, in1, a0, a1, cell_index);
    # Pol offset
    pol_offset = 2*p1 + p0;
    # Word index (in units of words (i.e. floats) of real component
    index = (cell_index * num_words_per_cell) + pol_offset;
    return index;

class Corr(Block):
    """
    **Functionality**

    This block reads data from a GPU-side bifrost ring buffer and feeds
    it to xGPU for correlation, outputing results to another GPU-side buffer.

    **New Sequence Condition**

    This block starts a new sequence each time a new integration
    configuration is loaded or the upstream sequence changes.

    **Input Header Requirements**

    This block requires that the following header fields
    be provided by the upstream data source:

    .. table::
        :widths: 25 10 10 55

        +-----------+--------+-------+------------------------------------------------+
        | Field     | Format | Units | Description                                    |
        +===========+========+=======+================================================+
        | ``seq0``  | int    |       | Spectra number for the first sample in the     |
        |           |        |       | input sequence                                 |
        +-----------+--------+-------+------------------------------------------------+

    **Output Headers**

    This block passes headers from the upstream block with
    the following modifications:

    .. table::
        :widths: 25 10 10 55

        +------------------+----------------+---------+-------------------------------+
        | Field            | Format         | Units   | Description                   |
        +==================+================+=========+===============================+
        | ``seq0``         | int            |         | Spectra number for the        |
        |                  |                |         | *first* sample in the         |
        |                  |                |         | integrated output             |
        +------------------+----------------+---------+-------------------------------+
        | ``acc_len``      | int            |         | Number of spectra integrated  |
        |                  |                |         | into each output sample by    |
        |                  |                |         | this block                    |
        +------------------+----------------+---------+-------------------------------+
        | ``ant_to_input`` | list of ints   |         | This header is removed from   |
        |                  |                |         | the sequence                  |
        +------------------+----------------+---------+-------------------------------+
        | ``input_to_ant`` | list of ints   |         | This header is removed from   |
        |                  |                |         | the sequence                  |
        +------------------+----------------+---------+-------------------------------+

    **Data Buffers**

    *Input Data Buffer*: A GPU-side bifrost ring buffer of 4+4 bit
    complex data in order: ``time x channel x stand x polarization``.

    Each gulp of the input buffer reads ``ntime_gulp`` samples,
    which should match *both* the xGPU
    compile-time parameters ``NTIME`` and ``NTIME_PIPE``.

    *Output Data Buffer*: A GPU-side bifrost ring buffer of 32+32 bit complex
    integer data. This buffer is in the xGPU triangular matrix order:
    ``time x channel x complexity x baseline``.

    The output buffer is written in single accumulation blocks (an integration of
    ``acc_len`` input time samples).

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
        the xGPU ``NFREQUENCY`` compile-time parameter.
    :type nchan: int

    :param nstand: Number of stands per time sample. This should match
        the xGPU ``NSTATION`` compile-time parameter.
    :type nstand: int

    :param npol: Number of polarizations per stand. This should match
       the xGPU ``NPOL`` compile-time parameter.
    :type npol: int

    :param acc_len: Accumulation length per output buffer write. This should
        be an integer multiple of the input gulp size ``ntime_gulp``.
        This parameter can be updated at runtime.
    :type acc_len: int

    :parameter etcd_client: Etcd client object used to facilitate control of this block.
        If ``None``, do not use runtime control.
    :type etcd_client: etcd3.client.Etcd3Client

    :parameter test: If True, run a CPU correlator in parallel with xGPU and
        verify the output. Beware, the (Python!) CPU correlator is *very* slow.
    :type test: Bool

    :parameter autostartat: The start time at which the correlator should
        automatically being correlating without intervention of the runtime control
        system. Use the value ``-1`` to cause integration to being on the next
        gulp.
    :type autostartat: int

    :parameter ant_to_input: an [nstand, npol] list of input IDs used to map
        stand/polarization ``S``, ``P`` to a correlator input. This allows the block
        to pass this information to downstream processors. *This functionality is
        currently unused*
    :type ant_to_input: nstand x npol list of ints

    **Runtime Control and Monitoring**

    This block accepts the following command fields:

    .. table::
        :widths: 25 10 10 55

        +-----------------+--------+---------+------------------------------+
        | Field           | Format | Units   | Description                  |
        +=================+========+=========+==============================+
        | ``acc_len``     | int    | samples | Number of samples to         |
        |                 |        |         | accumulate. This should be a |
        |                 |        |         | multiple of ``ntime_gulp``   |
        +-----------------+--------+---------+------------------------------+
        | ``start_time``  | int    | samples | The desired first time       |
        |                 |        |         | sample in an accumulation.   |
        |                 |        |         | This should be a multiple of |
        |                 |        |         | ``ntime_gulp``, and should   |
        |                 |        |         | be related to GPS time       |
        |                 |        |         | through external knowledge   |
        |                 |        |         | of the spectra count origin  |
        |                 |        |         | (aka SNAP *sync time*). The  |
        |                 |        |         | special value ``-1`` can be  |
        |                 |        |         | used to force an immediate   |
        |                 |        |         | start of the correlator on   |
        |                 |        |         | the next input gulp.         |
        +-----------------+--------+---------+------------------------------+

    """

    def __init__(self, log, iring, oring, ntime_gulp=2500,
                 guarantee=True, core=-1, nchan=192, npol=2, nstand=352, acc_len=2400, gpu=-1, test=False, etcd_client=None, autostartat=0, ant_to_input=None):
        assert (acc_len % ntime_gulp == 0), "Acculmulation length must be a multiple of gulp size"
        super(Corr, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        self.ntime_gulp = ntime_gulp
        self.nchan = nchan
        self.npol = npol
        self.nstand = nstand
        self.matlen = nchan * (nstand//2+1)*(nstand//4)*npol*npol*4
        self.gpu = gpu

        self.test = test

        if self.gpu != -1:
            BFSetGPU(self.gpu)
        
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nchan*nstand*npol*1        # complex8
        self.ogulp_size = self.matlen * 8 # complex64

        self.define_command_key('start_time', type=int, initial_val=autostartat,
                                condition=lambda x: (x == -1) or (x % self.ntime_gulp == 0))
        self.define_command_key('acc_len', type=int, initial_val=acc_len,
                                condition=lambda x: x % self.ntime_gulp == 0)
        self.update_stats({'xgpu_acc_len': self.ntime_gulp})

        # initialize xGPU. Arrays passed as inputs don't really matter here
        # but we need to pass something
        ibuf = BFArray([0], dtype='i8', space='cuda')
        obuf = BFArray([0], dtype='i64', space='cuda')
        rv = _bf.bfXgpuInitialize(ibuf.as_BFarray(), obuf.as_BFarray(), self.gpu)
        if (rv != _bf.BF_STATUS_SUCCESS):
            self.log.error("xgpuIntialize returned %d" % rv)
            raise RuntimeError

        # generate xGPU order map
        self.antpol_to_input = BFArray(np.zeros([nstand, npol], dtype=np.int32), space='system')
        self.antpol_to_bl = BFArray(np.zeros([nstand, nstand, npol, npol], dtype=np.int32), space='system')
        self.bl_is_conj   = BFArray(np.zeros([nstand, nstand, npol, npol], dtype=np.int32), space='system')
        if ant_to_input is not None:
            self.update_baseline_indices(ant_to_input)

    def _test(self, din, nchan, nstand, npol):
        din_cpu = din.copy(space='system')
        # TODO all this casting seems like it is superfluous, but
        # instructions like `dr[dr>7] -= 16` don't behave as [JH] expected
        # when called against raw views.
        d = np.array(din_cpu.view(dtype=np.uint8).reshape([self.ntime_gulp, nchan, nstand, npol]))
        dr = np.array(d >> 4, dtype=np.int8)
        dr[dr>7] -= 16
        di = np.array(d & 0xf, dtype=np.int8)
        di[di>7] -= 16
        dc = dr + 1j*di
        out = np.zeros([nchan, nstand, nstand, npol*npol], dtype=np.complex)
        print("Computing correlation on CPU")
        tick = time.time()
        for t in range(self.ntime_gulp):
            for chan in range(nchan):
                out[chan, :, :, 0] += np.outer(np.conj(dc[t,chan,:,0]), dc[t,chan,:,0])
                out[chan, :, :, 1] += np.outer(np.conj(dc[t,chan,:,0]), dc[t,chan,:,1])
                out[chan, :, :, 2] += np.outer(np.conj(dc[t,chan,:,1]), dc[t,chan,:,0])
                out[chan, :, :, 3] += np.outer(np.conj(dc[t,chan,:,1]), dc[t,chan,:,1])
        tock = time.time()
        print("CPU correlation took %d seconds" % (tock - tick))
        return out

    def _compare(self, din, dout, nchan, nstand, npol):
        print("Copying GPU results to CPU")
        dout_cpu = dout.copy(space='system')
        dout_cpu_reshape = dout_cpu.view(dtype='i32').reshape([2, nchan, 47849472//nchan])
        dout_c = dout_cpu_reshape[0,:,:] + 1j*dout_cpu_reshape[1,:,:]
        # Generate a more logically ordered output
        print("Reordering GPU results")
        tick = time.time()
        gpu_reorder = np.zeros([nchan, nstand, nstand, npol*npol], dtype=np.complex)
        for chan in range(nchan):
            for i0 in range(nstand):
                for i1 in range(i0, nstand):
                    for p in range(4):
                        p0 = p // 2
                        p1 = p % 2
                        v = dout_c[chan, regtile_index(2*i0+p0, 2*i1+p1, nstand)] # only valuid for i1>=i0
                        gpu_reorder[chan, i0, i1, p] = v
        tock = time.time()
        print("Reordering took %d seconds" % (tock - tick))
        
        print("Comparing with CPU calculation.")
        ok = True
        for chan in range(nchan):
            for i0 in range(nstand):
                for i1 in range(i0, nstand):
                    ok = ok and np.all(din[0,i0,i1,:]==gpu_reorder[0,i0,i1,:])
        print("MATCH?", ok)

    def update_baseline_indices(self, ant_to_input):
        """
        Using a map of stand,pol -> correlator ID, create an array mapping
        stand0,pol0 * stand1,pol1 -> baseline ID
        Inputs:
            ant_to_input : [nstand x npol] list of input IDs
        Outputs:
            [nstand x nstand x npol x npol] list of baseline_index, is_conjugated

            where baseline_index is the index within an xGPU buffer, and is_conjugated
            is 1 if the data should be conjugated, or 0 otherwise.
        """
        cpu_affinity.set_core(self.core)
        self.antpol_to_input[...] = ant_to_input
        _bf.bfXgpuGetOrder(self.antpol_to_input.as_BFarray(),
                           self.antpol_to_bl.as_BFarray(),
                           self.bl_is_conj.as_BFarray())

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        time_tag = 1
        self.update_stats({'state': 'starting'})
        with self.oring.begin_writing() as oring:
            prev_time = time.time()
            # Don't start when code begins. But, don't necessarily stop on each new sequence
            start = False
            self.update_pending = True
            for iseq in self.iring.read(guarantee=self.guarantee):
                self.log.info('CORR >> new input sequence!')
                process_time = 0
                oseq = None
                ospan = None
                ihdr = json.loads(iseq.header.tostring())
                this_gulp_time = ihdr['seq0']
                ohdr = ihdr.copy()
                # If the correlator was running before and a new sequence came
                # try to realign to an appropriate integration boundary and continue
                if start:
                    last_start_time = start_time
                    # number of integrations missed
                    missed_time = (this_gulp_time - last_start_time)
                    missed_accs = missed_time // acc_len
                    # New start time
                    start_time = last_start_time + (missed_accs + 10)*acc_len
                    # Don't start until we get to this time
                    start = False
                    self.log.info("CORR >> Recovering start time set to %d. Accumulating %d samples" % (start_time, acc_len))
                    ohdr['acc_len'] = acc_len
                    ohdr['seq0'] = start_time

                # Uncomment if you want the baseline order to be computed on the fly
                # self.update_baseline_indices(ihdr['ant_to_input'])

                # Remove ant-to-input maps. This block outputs xGPU-formatted data,
                # Which isn't trivially an nstand/npol x nstand/npol array.
                # Assume that the downstream code knows how the baseline list is formatted.
                # It would be nice to put that information in the header, but this seems
                # to cause unexpectedly severe slowdown
                if 'ant_to_input' in ihdr:
                    ohdr.pop('ant_to_input')
                if 'input_to_ant' in ihdr:
                    ohdr.pop('input_to_ant')
                self.sequence_proclog.update(ohdr)
                # uncomment if you want downstream processors to deal with input ordering on the fly
                # ohdr.update({'ant_to_bl_id': self.antpol_to_bl.tolist(), 'bl_is_conj': self.bl_is_conj.tolist()})
                for ispan in iseq.read(self.igulp_size):
                    if ispan.size < self.igulp_size:
                        self.log.info("CORR >>> Ignoring final gulp (expected %d bytes but got %d)" % (self.igulp_size, ispan.size))
                        continue # ignore final gulp
                    if self.update_pending:
                        self.update_command_vals()
                        # Use start_time = -1 as a special condition to start on the next sample
                        # which is a multiple of the accumulation length
                        acc_len = self.command_vals['acc_len']
                        if self.command_vals['start_time'] == -1:
                            start_time = (this_gulp_time - (this_gulp_time % acc_len) + acc_len)
                        else:
                            start_time = self.command_vals['start_time']
                        start = False
                        self.log.info("CORR >> New start time set to %d. Accumulating %d samples" % (start_time, acc_len))
                        ohdr['acc_len'] = acc_len
                        ohdr['seq0'] = start_time
                    self.update_stats({'curr_sample': this_gulp_time})
                    # If this is the start time, update the first flag, and compute where the last flag should be
                    if this_gulp_time == start_time:
                        self.log.info("CORR >> Start time %d reached." % start_time)
                        start = True
                        first = start_time
                        last  = first + acc_len - self.ntime_gulp
                        # on a new accumulation start, if a current oseq is open, close it, and start afresh
                        if oseq: oseq.end()
                        ohdr_str = json.dumps(ohdr)
                        self.sequence_proclog.update(ohdr)
                        oseq = oring.begin_sequence(time_tag=time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        time_tag += 1
                    if not start:
                        self.update_stats({'state': 'waiting'})
                        this_gulp_time += self.ntime_gulp
                        continue
                    self.update_stats({'state': 'running'})
                    # Use acc_len = 0 as a special stop condition
                    if acc_len == 0:
                        self.update_stats({'state': 'stopped'})
                        if oseq: oseq.end()
                        oseq = None
                        start = False

                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    if this_gulp_time == first:
                        # reserve an output span
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                        if self.test:
                            test_out = np.zeros([ihdr['nchan'], ihdr['nstand'], ihdr['nstand'], ihdr['npol']**2], dtype=np.complex)
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                    if not ospan:
                        self.log.error("CORR: trying to write to not-yet-opened ospan")
                    if self.test:
                        test_out += self._test(ispan.data, ihdr['nchan'], ihdr['nstand'], ihdr['npol'])
                    _bf.bfXgpuKernel(ispan.data.as_BFarray(), ospan.data.as_BFarray(), int(this_gulp_time==last))
                    curr_time = time.time()
                    process_time += curr_time - prev_time
                    prev_time = curr_time
                    if this_gulp_time == last:
                        if self.test:
                            self._compare(test_out, ospan.data, ihdr['nchan'], ihdr['nstand'], ihdr['npol'])
                        ospan.close()
                        throughput_gbps = 8 * acc_len * ihdr['nchan'] * ihdr['nstand'] * ihdr['npol'] / process_time / 1e9
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': throughput_gbps})
                        self.update_stats({'last_end_sample': this_gulp_time, 'throughput': throughput_gbps})
                        process_time = 0
                        # Update integration boundary markers
                        first = last + self.ntime_gulp
                        last = first + acc_len - self.ntime_gulp
                    # And, update overall time counter
                    this_gulp_time += self.ntime_gulp
                if oseq: oseq.end()
                oseq = None
                start = False
                            
            # If upstream process stops producing, close things gracefully
            # TODO: why is this necessary? Get exceptions from ospan.__exit__ if not here
            if oseq:
                ospan.close()
                oseq.end()
