import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.device import set_device as BFSetGPU

import time
import simplejson as json
import numpy as np

from blocks.block_base import Block

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
    Perform cross-correlation using xGPU
    """
    def __init__(self, log, iring, oring, ntime_gulp=2500,
                 guarantee=True, core=-1, nchans=192, npols=2, nstands=352, acc_len=2400, gpu=-1, test=False, etcd_client=None, autostartat=0):
        assert (acc_len % ntime_gulp == 0), "Acculmulation length must be a multiple of gulp size"
        super(Corr, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        self.ntime_gulp = ntime_gulp
        self.nchans = nchans
        self.npols = npols
        self.nstands = nstands
        self.matlen = nchans * (nstands//2+1)*(nstands//4)*npols*npols*4
        self.gpu = gpu

        self.test = test

        if self.gpu != -1:
            BFSetGPU(self.gpu)
        
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nchans*nstands*npols*1        # complex8
        self.ogulp_size = self.matlen * 8 # complex64

        self.new_start_time = autostartat
        self.new_acc_len = 2400
        self.update_pending=True
        self.stats_proclog.update({'new_acc_len': self.new_acc_len,
                                   'new_start_sample': self.new_start_time,
                                   'update_pending': self.update_pending,
                                   'last_cmd_time': time.time()})

        # initialize xGPU. Arrays passed as inputs don't really matter here
        # but we need to pass something
        ibuf = BFArray(0, dtype='i8', space='cuda')
        obuf = BFArray(0, dtype='i64', space='cuda')
        rv = _bf.bfXgpuInitialize(ibuf.as_BFarray(), obuf.as_BFarray(), self.gpu)
        if (rv != _bf.BF_STATUS_SUCCESS):
            self.log.error("xgpuIntialize returned %d" % rv)
            raise RuntimeError

        # generate xGPU order map
        self.antpol_to_input = BFArray(np.zeros([nstands, npols], dtype=np.int32), space='system')
        self.antpol_to_bl = BFArray(np.zeros([nstands, nstands, npols, npols], dtype=np.int32), space='system')
        self.bl_is_conj   = BFArray(np.zeros([nstands, nstands, npols, npols], dtype=np.int32), space='system')

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

    def _etcd_callback(self, watchresponse):
        """
        A callback to run whenever this block's command key is updated.
        Decodes integration start time and accumulation length and
        preps to update the pipeline at the end of the next integration.
        """
        v = json.loads(watchresponse.events[0].value)
        if 'acc_len' not in v or not isinstance(v['acc_len'], int):
            self.log.error("CORR: Incorrect or missing acc_len")
            return
        if 'start_time' not in v or not isinstance(v['start_time'], int):
            self.log.error("CORR: Incorrect or missing start_time")
            return
        if v['acc_len'] % self.ntime_gulp != 0:
            self.log.error("CORR: Acc length must be a multiple of %d" % self.ntime_gulp)
            return
        if v['start_time'] % self.ntime_gulp != 0:
            self.log.error("CORR: Start time must be a multiple of %d" % self.ntime_gulp)
            return
        self.acquire_control_lock()
        self.new_start_time = v['start_time']
        self.new_acc_len = v['acc_len']
        self.update_pending = True
        self.stats_proclog.update({'new_acc_len': self.new_acc_len,
                                   'new_start_sample': self.new_start_time,
                                   'update_pending': self.update_pending,
                                   'last_cmd_time': time.time()})
        self.release_control_lock()

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
        self.antpol_to_input[...] = ant_to_input
        _bf.bfXgpuGetOrder(self.antpol_to_input.as_BFarray(),
                           self.antpol_to_bl.as_BFarray(),
                           self.bl_is_conj.as_BFarray())
        return self.antpol_to_bl.tolist(), self.bl_is_conj.tolist()
        
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        oseq = None
        ospan = None
        start = False
        with self.oring.begin_writing() as oring:
            prev_time = time.time()
            for iseq in self.iring.read(guarantee=self.guarantee):
                self.log.debug("Correlating output")
                ihdr = json.loads(iseq.header.tostring())
                this_gulp_time = ihdr['seq0']
                ohdr = ihdr.copy()
                ohdr.pop('ant_to_input')
                ohdr.pop('input_to_ant')
                self.sequence_proclog.update(ohdr)
                antpol_to_bl_id, bl_is_conj = self.update_baseline_indices(ihdr['ant_to_input'])
                ohdr.update({'ant_to_bl_id': antpol_to_bl_id, 'bl_is_conj': bl_is_conj})
                for ispan in iseq.read(self.igulp_size):
                    if self.update_pending:
                        self.acquire_control_lock()
                        start_time = self.new_start_time
                        acc_len = self.new_acc_len
                        start = False
                        self.log.info("CORR >> Starting at %d. Accumulating %d samples" % (self.new_start_time, self.new_acc_len))
                        self.update_pending = False
                        self.release_control_lock()
                        ohdr['acc_len'] = acc_len
                        ohdr['start_time'] = start_time
                        self.stats_proclog.update({'acc_len': acc_len,
                                                   'start_sample': start_time,
                                                   'curr_sample': this_gulp_time,
                                                   'update_pending': self.update_pending,
                                                   'last_update_time': time.time()})
                    self.stats_proclog.update({'curr_sample': this_gulp_time})
                    # If this is the start time, update the first flag, and compute where the last flag should be
                    if this_gulp_time == start_time:
                        start = True
                        first = start_time
                        last  = first + acc_len - self.ntime_gulp
                        # on a new accumulation start, if a current oseq is open, close it, and start afresh
                        if oseq: oseq.end()
                        ohdr_str = json.dumps(ohdr)
                        oseq = oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                    if not start:
                        this_gulp_time += self.ntime_gulp
                        continue
                    # Use acc_len = 0 as a special stop condition
                    if acc_len == 0:
                        if oseq: oseq.end()
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
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    if this_gulp_time == last:
                        if self.test:
                            self._compare(test_out, ospan.data, ihdr['nchan'], ihdr['nstand'], ihdr['npol'])
                        ospan.close()
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': 8*self.igulp_size / process_time / 1e9})
                        self.stats_proclog.update({'last_end_sample': this_gulp_time})
                        process_time = 0
                        # Update integration boundary markers
                        first = last + self.ntime_gulp
                        last = first + acc_len - self.ntime_gulp
                    # And, update overall time counter
                    this_gulp_time += self.ntime_gulp
                            
            # If upstream process stops producing, close things gracefully
            # TODO: why is this necessary? Get exceptions from ospan.__exit__ if not here
            if oseq:
                ospan.close()
                oseq.end()

