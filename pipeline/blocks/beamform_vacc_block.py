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
import socket

from blocks.block_base import Block

class BeamVacc(Block):
    """
    Copy data from one buffer to another.
    """
    def __init__(self, log, iring, oring, ninput_beam=16, beam_id=0, gpu=0, autostartat=-1, acc_len=24000,
                 guarantee=True, core=-1, nchan=192, etcd_client=None):

        super(BeamVacc, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)
        self.ninput_beam = ninput_beam
        self.nchan = nchan
        self.beam_id = beam_id

        self.gpu = gpu

        if self.gpu != -1:
            BFSetGPU(self.gpu)

        self.ogulp_size = nchan*4*4  # XX, YY, real(XY), im(XY) x 32-bit float
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Accumulator for a single beam
        self.accdata = BFArray(np.zeros([nchan, 4]), dtype='f32', space='cuda')
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
        ospan = None
        oseq = None
        start = False
        process_time = 0
        time_tag = 1
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                self.update_pending = True
                ihdr = json.loads(iseq.header.tostring())
                ntime_block = 1#ihdr['ntime_block']
                nbeam = ihdr['nbeam']
                nchan = ihdr['nchan']
                igulp_size = nbeam * ntime_block * nchan * 4 * 4 #XX, YY, XYr, XYi, f32
                this_gulp_time = ihdr['seq0']
                ohdr = ihdr.copy()
                # Mash header in here if you want
                upstream_acc_len = ihdr['acc_len']
                ohdr['upstream_acc_len'] = upstream_acc_len
                ohdr_str = json.dumps(ohdr)
                prev_time = time.time()
                for ispan in iseq.read(igulp_size):
                    if self.update_pending:
                        self.acquire_control_lock()
                        start_time = self.new_start_time
                        acc_len = self.new_acc_len
                        # Use start_time = -1 as a special condition to start on the next sample
                        if start_time == -1:
                            start_time = this_gulp_time
                        start = False
                        self.log.info("BEAMACC%d >> New start time at %d. Accumulation: %d samples" % (self.beam_id, self.new_start_time, self.new_acc_len))
                        self.update_pending = False
                        self.stats.update({'acc_len': acc_len,
                                           'start_sample': start_time,
                                           'curr_sample': this_gulp_time,
                                           'update_pending': self.update_pending,
                                           'last_update_time': time.time()})
                        self.update_stats()
                        self.release_control_lock()
                        if acc_len % upstream_acc_len != 0:
                            self.log.error("BEAMACC%d >> Requested acc_len %d incompatible with upstream integration %d" % (self.beam_id, acc_len, upstream_acc_len))
                        #if acc_len != 0 and ((start_time - upstream_start_time) % upstream_acc_len != 0):
                        #    self.log.error("BEAMACC%d >> Requested start_time %d incompatible with upstream integration %d" % (self.beam_id, acc_len, upstream_acc_len))
                        ohdr['acc_len'] = acc_len
                        ohdr['seq0'] = start_time
                    self.stats.update({'curr_sample': this_gulp_time})
                    self.update_stats()
                    # Use acc_len = 0 as a special stop condition
                    if acc_len == 0:
                        if oseq: oseq.end()
                        start = False
                        this_gulp_time += upstream_acc_len * ntime_block
                        continue

                    # If we get here, acc_len is != 0, and we are searching for
                    # a new integration boundary

                    # If this is the start time, update the first flag, and compute where the last flag should be
                    if this_gulp_time == start_time:
                        start = True
                        first = start_time
                        last  = first + acc_len - upstream_acc_len*ntime_block
                        # on a new accumulation start, if a current oseq is open, close it, and start afresh
                        if oseq: oseq.end()
                        ohdr_str = json.dumps(ohdr)
                        self.sequence_proclog.update(ohdr)
                        oseq = oring.begin_sequence(time_tag=time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        time_tag += 1
                        self.log.info("BEAMACC%d >> Start time %d reached. Accumulating to %d (upstream accumulation: %d)" % (self.beam_id, start_time, last, upstream_acc_len))

                    # If we're still waiting for a start, spin the wheels
                    if not start:
                        this_gulp_time += upstream_acc_len * ntime_block
                        continue

                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    self.log.debug("Accumulating beam %d" % self.beam_id)
                    idata = ispan.data_view('f32').reshape([nbeam, ntime_block, nchan, 4])[self.beam_id, 0]
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
                        self.log.info("BEAMACC%d > Last accumulation input" % self.beam_id)
                        # copy to CPU
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                        odata = ospan.data_view('f32').reshape(self.accdata.shape)
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
                        first = last + upstream_acc_len*ntime_block
                        last = first + acc_len - upstream_acc_len*ntime_block
                    # And, update overall time counter
                    this_gulp_time += upstream_acc_len * ntime_block
            # If upstream process stops producing, close things gracefully
            # TODO: why is this necessary? Get exceptions from ospan.__exit__ if not here
            if ospan: ospan.close()
            if oseq: oseq.end()
