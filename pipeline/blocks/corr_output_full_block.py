import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap
from bifrost.ndarray import copy_array
from bifrost.device import stream_synchronize, set_device as BFSetGPU

import os
import time
import simplejson as json
import socket
import struct
import numpy as np

from blocks.block_base import Block

class CorrOutputFull(Block):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring,
                 guarantee=True, core=-1, nchans=192, npols=2, nstands=352, etcd_client=None, dest_port=10000,
                 checkfile=None, checkfile_acc_len=1):
        # TODO: Other things we could check:
        # - that nchans/pols/gulp_size matches XGPU compilation
        super(CorrOutputFull, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        self.nchans = nchans
        self.npols = npols
        self.nstands = nstands
        self.matlen = nchans * (nstands//2+1)*(nstands//4)*npols*npols*4

        self.igulp_size = self.matlen * 8 # complex64

        # Arrays to hold the conjugation and bl indices of data coming from xGPU
        self.antpol_to_bl = BFArray(np.zeros([nstands, nstands, npols, npols], dtype=np.int32), space='system')
        self.bl_is_conj   = BFArray(np.zeros([nstands, nstands, npols, npols], dtype=np.int32), space='system')
        self.reordered_data = BFArray(np.zeros([nstands, nstands, npols, npols, nchans, 2], dtype=np.int32), space='system')

        self.checkfile_acc_len = checkfile_acc_len
        if checkfile is None:
            self.checkfile = None
        else:
            self.checkfile = open(checkfile, 'rb')
            self.checkfile_nbytes = os.path.getsize(checkfile)
            self.log.info("CORR OUTPUT >> Checkfile %s" % self.checkfile.name)
            self.log.info("CORR OUTPUT >> Checkfile length: %d bytes" % self.checkfile_nbytes)
            self.log.info("CORR OUTPUT >> Checkfile accumulation length: %d" % self.checkfile_acc_len)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.dest_ip = None
        self.new_dest_ip = None
        self.dest_port = dest_port
        self.new_dest_port = dest_port
        self.update_pending = True

    def _etcd_callback(self, watchresponse):
        """
        A callback to run whenever this block's command key is updated.
        Decodes integration start time and accumulation length and
        preps to update the pipeline at the end of the next integration.
        """
        v = json.loads(watchresponse.events[0].value)
        if 'dest_ip' in v:
            self.new_dest_ip = v['dest_ip']
        if 'dest_port' in v:
            self.new_dest_port = v['dest_port']
        self.update_pending = True
        self.stats_proclog.update({'new_dest_ip': self.new_dest_ip,
                                   'new_dest_port': self.new_dest_port,
                                   'update_pending': self.update_pending,
                                   'last_cmd_time': time.time()})

    def get_checkfile_corr(self, t):
        """
        Get a single integration from the test file,
        looping back to the beginning of the file when
        the end is reached.
        Inputs: t (int) -- time index of correlation
        """
        dim = np.array([self.nchans, self.nstands, self.nstands, self.npols, self.npols])
        nbytes = dim.prod() * 2 * 8
        seekloc = (nbytes * t) % self.checkfile_nbytes
        self.log.debug("CORR OUTPUT >> Testfile has %d bytes. Seeking to %d and reading %d bytes for sample %d" % (self.checkfile_nbytes, seekloc, nbytes, t))
        self.checkfile.seek(seekloc)
        dtest_raw = self.checkfile.read(nbytes)
        if len(dtest_raw) != nbytes:
            self.log.error("CORR OUTPUT >> Failed to get correlation matrix from checkfile")
            return np.zeros(dim, dtype=np.complex)
        return np.frombuffer(dtest_raw, dtype=np.complex).reshape(dim)

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        prev_time = time.time()
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            upstream_acc_len = ihdr['acc_len']
            upstream_start_time = ihdr['start_time']
            self.antpol_to_bl[...] = ihdr['ant_to_bl_id']
            self.bl_is_conj[...] = ihdr['bl_is_conj']
            for ispan in iseq.read(self.igulp_size):
                # Update destinations if necessary
                if self.update_pending:
                    self.dest_ip = self.new_dest_ip
                    self.dest_port = self.new_dest_port
                    self.update_pending = False
                    self.log.info("CORR OUTPUT >> Updating destination to %s:%s" % (self.dest_ip, self.dest_port))
                    self.stats_proclog.update({'dest_ip': self.dest_ip,
                                               'dest_port': self.dest_port,
                                               'update_pending': self.update_pending,
                                               'last_update_time': time.time()})
                self.stats_proclog.update({'curr_sample': this_gulp_time})
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                _bf.bfXgpuReorder(ispan.data.as_BFarray(), self.reordered_data.as_BFarray(), self.antpol_to_bl.as_BFarray(), self.bl_is_conj.as_BFarray())
                # Check against test data if a file is provided
                if self.checkfile:
                    assert upstream_acc_len % self.checkfile_acc_len == 0, "CORR OUTPUT >> Testfile acc len not compatible with pipeline acc len"
                    assert upstream_start_time % self.checkfile_acc_len == 0, "CORR OUTPUT >> Testfile acc len not compatible with pipeline start time"
                    nblocks = (upstream_acc_len // self.checkfile_acc_len)
                    self.log.info("CORR OUTPUT >> Computing expected output from test file")
                    self.log.info("CORR OUTPUT >> Upstream accumulation %d" % upstream_acc_len)
                    self.log.info("CORR OUTPUT >> File accumulation %d" % self.checkfile_acc_len)
                    self.log.info("CORR OUTPUT >> Integrating %d blocks" % nblocks)
                    dtest = np.zeros([self.nchans, self.nstands, self.nstands, self.npols, self.npols], dtype=np.complex)
                    for i in range(nblocks):
                        dtest += self.get_checkfile_corr(this_gulp_time // self.checkfile_acc_len + i)
                    # check baseline by baseline
                    badcnt = 0
                    goodcnt = 0
                    nonzerocnt = 0
                    zerocnt = 0
                    now = time.time()
                    for s0 in range(self.nstands):
                        if time.time() - now > 15:
                            self.log.info("CORR OUTPUT >> Check complete for stand %d" % s0)
                            now = time.time()
                        for s1 in range(s0, self.nstands):
                            for p0 in range(self.npols):
                               for p1 in range(self.npols):
                                   if not np.all(self.reordered_data[s0, s1, p0, p1, :, 0] == 0):
                                       nonzerocnt += 1
                                   else:
                                       zerocnt += 1
                                   if not np.all(self.reordered_data[s0, s1, p0, p1, :, 1] == 0):
                                       nonzerocnt += 1
                                   else:
                                       zerocnt += 1
                                   if np.any(self.reordered_data[s0, s1, p0, p1, :, 0] != dtest[:, s0, s1, p0, p1].real):
                                       self.log.error("CORR OUTPUT >> test vector mismatch! [%d, %d, %d, %d] real" %(s0,s1,p0,p1))
                                       print("antpol to bl: %d" % self.antpol_to_bl[s0,s1,p0,p1])
                                       print("is conjugated : %d" % self.bl_is_conj[s0,s1,p0,p1])
                                       print("pipeline:", self.reordered_data[s0, s1, p0, p1, 0:5, 0])
                                       print("expected:", dtest[0:5, s0, s1, p0, p1].real)
                                       badcnt += 1
                                   else:
                                       goodcnt += 1
                                   if np.any(self.reordered_data[s0, s1, p0, p1, :, 1] != dtest[:, s0, s1, p0, p1].imag): # test data follows inverse conj convention
                                       self.log.error("CORR OUTPUT >> test vector mismatch! [%d, %d, %d, %d] imag" %(s0,s1,p0,p1))
                                       print("antpol to bl: %d" % self.antpol_to_bl[s0,s1,p0,p1])
                                       print("is conjugated : %d" % self.bl_is_conj[s0,s1,p0,p1])
                                       print("pipeline:", self.reordered_data[s0, s1, p0, p1, 0:5, 1])
                                       print("expected:", dtest[0:5, s0, s1, p0, p1].imag)
                                       badcnt += 1
                                   else:
                                       goodcnt += 1
                    if badcnt > 0:
                        self.log.error("CORR OUTPUT >> test vector check complete. Good: %d, Bad: %d, Non-zero: %d, Zero: %d" % (goodcnt, badcnt, nonzerocnt, zerocnt))
                    else:
                        self.log.info("CORR OUTPUT >> test vector check complete. Good: %d, Bad: %d, Non-zero: %d, Zero: %d" % (goodcnt, badcnt, nonzerocnt, zerocnt))

                if self.dest_ip is not None:
                    dout = np.zeros([self.npols, self.npols, self.nchans, 2], dtype='>i')
                    for s0 in range(self.nstands):
                        for s1 in range(s0, self.nstands):
                            header = struct.pack(">Q3L", this_gulp_time, ihdr['chan0'], s0, s1)
                            dout = self.reordered_data[s0, s1]
                            self.sock.sendto(header + dout.tobytes(), (self.dest_ip, self.dest_port))
                    self.log.info("CORR OUPUT >> Sending complete")
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.stats_proclog.update({'last_end_sample': this_gulp_time})
                # And, update overall time counter
                this_gulp_time += upstream_acc_len
        if self.checkfile:
            self.checkfile.close()