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

from .block_base import Block

class CorrSubsel(Block):
    """
    Grab arbitrary entries from a GPU buffer and copy them to the CPU
    """
    nvis_out = 4656
    def __init__(self, log, iring, oring, guarantee=True, core=-1, etcd_client=None,
                 nchan=192, npol=2, nstand=352, nchan_sum=4, gpu=-1,
                 antpol_to_bl=None, bl_is_conj=None):

        super(CorrSubsel, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)

        self.nchan_in = nchan
        self.nchan_out = nchan // nchan_sum
        self.nchan_sum = nchan_sum
        self.npol = npol
        self.nstand = nstand
        self.gpu = gpu
        self.matlen = self.nchan_in * (nstand//2+1)*(nstand//4)*npol*npol*4 # xGPU defined

        if self.gpu != -1:
            BFSetGPU(self.gpu)

        self.igulp_size = self.matlen * 8 # complex64

        # Create an array of subselection indices on the GPU, and one on the CPU.
        # The user can update the CPU-side array, and the main processing thread
        # will copy this to the GPU when it changes
        # TODO: nvis_out could be dynamic, but we'd have to reallocate the GPU memory
        # if the size changed. Leave static for now, which is all the requirements call for.
        self._subsel      = BFArray(shape=[self.nvis_out], dtype='i32', space='cuda')
        self._subsel_next = BFArray(shape=[self.nvis_out], dtype='i32', space='cuda_host')
        self._baselines = np.zeros([self.nvis_out, 2, 2], dtype=np.int32)
        self._baselines_next = np.zeros([self.nvis_out, 2, 2], dtype=np.int32)
        self._conj      = BFArray(shape=[self.nvis_out], dtype='i32', space='cuda')
        self._conj_next = BFArray(shape=[self.nvis_out], dtype='i32', space='cuda_host')

        self.obuf_gpu = BFArray(shape=[self.nchan_out, self.nvis_out], dtype='ci32', space='cuda')
        self.ogulp_size = self.nchan_out * self.nvis_out * 8
        self.update_stats()
        if antpol_to_bl is not None:
            self._antpol_to_bl = antpol_to_bl
        else:
            self._antpol_to_bl = np.zeros([nstand, npol, nstand, npol])
        if bl_is_conj is not None:
            self._bl_is_conj = bl_is_conj
        else:
            self._bl_is_conj = np.zeros([nstand, npol, nstand, npol])

        # update subselection map to a default initial value of
        # pol 0 autos
        # This can't be called until the bl_is_conj and antpol_to_bl maps have been set above
        self.update_subsel([[[i % nstand,0], [i % nstand,0]] for i in range(self.nvis_out)])
        
    def _etcd_callback(self, watchresponse):
        """
        A callback to run when the etcd baseline order key is updated.
        Json-decodes the value of the etcd key, and calls update_subsel with
        this list of baseline indices.
        """
        v = json.loads(watchresponse.events[0].value)
        if 'subsel' not in v or not isinstance(v['subsel'], list):
            self.log.error("CORR SUBSEL >> Incorrect or missing subsel")
        else:
            self.update_subsel(v['subsel'])

    def update_subsel(self, baselines):
        """
        Update the baseline index list which should be subselected.
        Updates are not applied immediately, but are transferred to the
        GPU at the end of the current data block.
        """
        if len(baselines) != self.nvis_out:
            self.log.error("Tried to update baseline subselection with an array of length %d" % len(baselines))
            return
        else:
            self.acquire_control_lock()
            self._baselines_next[...] = baselines
            for v in range(self.nvis_out):
                i0, i1 = baselines[v]
                s0, p0 = i0
                s1, p1 = i1
                # index as S0, S1, P0, P1
                self._subsel_next.data[v] = self._antpol_to_bl[s0, s1, p0, p1]
                self._conj_next.data[v] = self._bl_is_conj[s0, s1, p0, p1]
            self._subsel_pending = True
            self.stats.update({'new_subsel': self._subsel_next,
                               'new_baselines': self._baselines_next,
                               'update_pending': self._subsel_pending,
                               'last_cmd_time': time.time()})
            self.update_stats()
            self.release_control_lock()
            self.log.info("CORR SUBSEL >> New subselect map received")

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        oseq = None
        time_tag = 1
        with self.oring.begin_writing() as oring:
            prev_time = time.time()
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                this_gulp_time = ihdr['seq0']
                acc_len = ihdr['acc_len']
                # Uncomment this if you want to read the map on the fly
                #antpol_to_bl = ihdr['antpol_to_bl']
                ohdr = ihdr.copy()
                #ohdr.pop('antpol_to_bl')
                ohdr['nchan'] = ihdr['nchan'] // self.nchan_sum
                ohdr['nvis'] = self.nvis_out
                chan_width = ihdr['bw_hz'] / ihdr['nchan']
                ohdr['sfreq'] = (ihdr['sfreq'] + ((self.nchan_sum - 1) * chan_width)) / self.nchan_sum
                # On a start of sequence, always grab new subselection
                self.acquire_control_lock()
                self.log.info("Updating baseline subselection indices")
                # copy to GPU
                #self._subsel[...] = self._subsel_next
                #self._conj[...] = self._conj_next
                copy_array(self._subsel, self._subsel_next)
                copy_array(self._conj, self._conj_next)
                self._baselines[...] = self._baselines_next
                self.stats.update({'subsel': self._subsel_next,
                                   'baselines': self._baselines_next,
                                   'update_pending': False,
                                   'last_update_time': time.time()})
                ohdr['baselines'] = self._baselines.tolist()
                self.update_stats()
                self.release_control_lock()
                ohdr_str = json.dumps(ohdr)
                oseq = oring.begin_sequence(time_tag=time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                time_tag += 1
                for ispan in iseq.read(self.igulp_size):
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    self.log.debug("Grabbing subselection")
                    idata = ispan.data_view('ci32').reshape(self.matlen)
                    self.acquire_control_lock()
                    self._subsel_pending = False
                    self.release_control_lock()
                    with oseq.reserve(self.ogulp_size) as ospan:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        rv = _bf.bfXgpuSubSelect(idata.as_BFarray(), self.obuf_gpu.as_BFarray(), self._subsel.as_BFarray(), self._conj.as_BFarray(), self.nchan_sum)
                        if (rv != _bf.BF_STATUS_SUCCESS):
                            self.log.error("xgpuSubSelect returned %d" % rv)
                            raise RuntimeError
                        odata = ospan.data_view(dtype='ci32').reshape([self.nchan_out, self.nvis_out])
                        copy_array(odata, self.obuf_gpu)
                        # Wait for copy to complete before committing span
                        stream_synchronize()
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': reserve_time, 
                                              'process_time': process_time,
                                              'this_sample' : this_gulp_time})
                    # tick the sequence counter to the next integration
                    this_gulp_time += acc_len
                    # If a baseline change is pending start a new sequence
                    # with an updated header
                    if self._subsel_pending:
                        oseq.end()
                        self.acquire_control_lock()
                        self.log.info("Updating baseline subselection indices")
                        self._subsel[...] = self._subsel_next
                        ohdr['baselines'] = self._baselines.tolist()
                        self.stats.update({'subsel': self._subsel_next,
                                           'update_pending': False,
                                           'last_update_time': time.time()})
                        self.release_control_lock()
                        self.update_stats()
                        #update time tag based on what has already been processed
                        ohdr['seq0'] = this_gulp_time
                        ohdr_str = json.dumps(ohdr)
                        oseq = oring.begin_sequence(time_tag=time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        time_tag += 1
                # if the iseq ends
                oseq.end()