import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap
from bifrost.ndarray import copy_array
from bifrost.device import stream_synchronize, set_device as BFSetGPU

import time
import simplejson as json
import numpy as np

from blocks.block_base import Block

class CorrAcc(Block):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring, oring,
                 guarantee=True, core=-1, nchans=192, npols=2, nstands=352, acc_len=24000, gpu=-1, etcd_client=None, autostartat=0):
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        super(CorrAcc, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)
        self.nchans = nchans
        self.npols = npols
        self.nstands = nstands
        self.matlen = nchans * (nstands//2+1)*(nstands//4)*npols*npols*4
        self.gpu = gpu

        if self.gpu != -1:
            BFSetGPU(self.gpu)
        
        self.igulp_size = self.matlen * 8 # complex64
        self.ogulp_size = self.igulp_size
        # integration buffer
        self.accdata = BFArray(shape=(self.igulp_size // 4), dtype='i32', space='cuda')
        self.bfbf = LinAlg()

        self.new_start_time = autostartat
        self.new_acc_len = acc_len
        self.update_pending=True
        self.stats.update({'new_acc_len': self.new_acc_len,
                           'new_start_sample': self.new_start_time,
                           'update_pending': self.update_pending,
                           'last_cmd_time': time.time()})
        self.update_stats()

    def _etcd_callback(self, watchresponse):
        """
        A callback to run whenever this block's command key is updated.
        Decodes integration start time and accumulation length and
        preps to update the pipeline at the end of the next integration.
        """
        v = json.loads(watchresponse.events[0].value)
        self.acquire_control_lock()
        if 'start_time' in v and isinstance(v['start_time'], int):
            self.new_start_time = v['start_time']
        if 'acc_len' in v and isinstance(v['acc_len'], int):
            self.new_acc_len = v['acc_len']
        self.update_pending = True
        self.release_control_lock()
        self.stats.update({'new_acc_len': self.new_acc_len,
                           'new_start_sample': self.new_start_time,
                           'update_pending': self.update_pending,
                           'last_cmd_time': time.time()})
        self.update_stats()

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
        process_time = 0
        time_tag = 1
        with self.oring.begin_writing() as oring:
            prev_time = time.time()
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                ohdr = ihdr.copy()
                this_gulp_time = ihdr['seq0']
                upstream_acc_len = ihdr['acc_len']
                ohdr['upstream_acc_len'] = upstream_acc_len
                upstream_start_time = this_gulp_time
                self.sequence_proclog.update(ohdr)
                for ispan in iseq.read(self.igulp_size):
                    if self.update_pending:
                        self.acquire_control_lock()
                        start_time = self.new_start_time
                        acc_len = self.new_acc_len
                        # Use start_time = -1 as a special condition to start on the next sample
                        if start_time == -1:
                            start_time = this_gulp_time
                        start = False
                        self.log.info("CORRACC >> New start time at %d. Accumulation: %d samples" % (self.new_start_time, self.new_acc_len))
                        self.update_pending = False
                        self.stats.update({'acc_len': acc_len,
                                           'start_sample': start_time,
                                           'curr_sample': this_gulp_time,
                                           'update_pending': self.update_pending,
                                           'last_update_time': time.time()})
                        self.update_stats()
                        self.release_control_lock()
                        if acc_len % upstream_acc_len != 0:
                            self.log.error("CORRACC >> Requested acc_len %d incompatible with upstream integration %d" % (acc_len, upstream_acc_len))
                        if acc_len != 0 and ((start_time - upstream_start_time) % upstream_acc_len != 0):
                            self.log.error("CORRACC >> Requested start_time %d incompatible with upstream integration %d" % (acc_len, upstream_acc_len))
                        ohdr['acc_len'] = acc_len
                        ohdr['seq0'] = start_time
                    self.stats.update({'curr_sample': this_gulp_time})
                    self.update_stats()
                    # Use acc_len = 0 as a special stop condition
                    if acc_len == 0:
                        if oseq: oseq.end()
                        start = False
                        this_gulp_time += upstream_acc_len
                        continue

                    # If we get here, acc_len is != 0, and we are searching for
                    # a new integration boundary

                    # If this is the start time, update the first flag, and compute where the last flag should be
                    if this_gulp_time == start_time:
                        start = True
                        first = start_time
                        last  = first + acc_len - upstream_acc_len
                        # on a new accumulation start, if a current oseq is open, close it, and start afresh
                        if oseq: oseq.end()
                        ohdr_str = json.dumps(ohdr)
                        self.sequence_proclog.update(ohdr)
                        oseq = oring.begin_sequence(time_tag=time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        time_tag += 1
                        self.log.info("CORRACC >> Start time %d reached. Accumulating to %d (upstream accumulation: %d)" % (start_time, last, upstream_acc_len))

                    # If we're still waiting for a start, spin the wheels
                    if not start:
                        this_gulp_time += upstream_acc_len
                        continue

                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    self.log.debug("Accumulating correlation")
                    idata = ispan.data_view('i32')
                    if this_gulp_time == first:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        # TODO: surely there are more sensible ways to implement a vacc
                        BFMap("a = b", data={'a': self.accdata, 'b': idata})
                    else:
                        BFMap("a += b", data={'a': self.accdata, 'b': idata})
                    curr_time = time.time()
                    process_time += curr_time - prev_time
                    prev_time = curr_time
                    if this_gulp_time == last:
                        self.log.debug("CORRACC > Last accumulation input")
                        # copy to CPU
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                        odata = ospan.data_view('i32').reshape(self.accdata.shape)
                        copy_array(odata, self.accdata)
                        # Wait for copy to complete before committing span
                        stream_synchronize()
                        ospan.close()
                        ospan = None
                        curr_time = time.time()
                        process_time += curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,})
                        self.stats.update({'last_end_sample': this_gulp_time})
                        self.update_stats()
                        process_time = 0
                        # Update integration boundary markers
                        first = last + upstream_acc_len
                        last = first + acc_len - upstream_acc_len
                    # And, update overall time counter
                    this_gulp_time += upstream_acc_len
            # If upstream process stops producing, close things gracefully
            # TODO: why is this necessary? Get exceptions from ospan.__exit__ if not here
            if ospan: ospan.close()
            if oseq: oseq.end()
