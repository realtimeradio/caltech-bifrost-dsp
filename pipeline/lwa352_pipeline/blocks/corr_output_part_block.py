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
import socket
import struct
import numpy as np

from .block_base import Block


class CorrOutputPart(Block):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring,
                 guarantee=True, core=-1, etcd_client=None, dest_port=10001, nvis_per_packet=16):
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        super(CorrOutputPart, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)

        self.nvis_per_packet = nvis_per_packet

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.01)
        self.dest_ip = "0.0.0.0"
        self.new_dest_ip = "0.0.0.0"
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
        self.stats.update({'new_dest_ip': self.new_dest_ip,
                           'new_dest_port': self.new_dest_port,
                           'update_pending': self.update_pending,
                           'last_cmd_time': time.time()})
        self.update_stats()

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        prev_time = time.time()
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            upstream_acc_len = ihdr['acc_len']
            upstream_start_time = this_gulp_time
            baselines = np.array(ihdr['baselines'], dtype='>i')
            baselines_flat = baselines.flatten()
            nchan = ihdr['nchan']
            chan0 = ihdr['chan0']
            bw_hz = ihdr['bw_hz']
            sfreq = ihdr['sfreq']
            nvis  = ihdr['nvis']
            igulp_size = nvis * nchan * 8
            dout = np.zeros(shape=[nvis, nchan, 2], dtype='>i')
            for ispan in iseq.read(igulp_size):
                # Update destinations if necessary
                if self.update_pending:
                    self.dest_ip = self.new_dest_ip
                    self.dest_port = self.new_dest_port
                    self.update_pending = False
                    self.log.info("CORR PART OUTPUT >> Updating destination to %s:%s" % (self.dest_ip, self.dest_port))
                    self.stats.update({'dest_ip': self.dest_ip,
                                       'dest_port': self.dest_port,
                                       'update_pending': self.update_pending,
                                       'last_update_time': time.time()})
                self.stats.update({'curr_sample': this_gulp_time})
                self.update_stats()
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                if self.dest_ip != "0.0.0.0":
                    idata = ispan.data_view('i32').reshape([nchan, nvis, 2]).transpose([1,0,2]) # baseline x chan x complexity
                    dout[...] = idata;
                    for vn in range(len(baselines)//self.nvis_per_packet):
                        header = struct.pack(">QQ2d4I",
                                             ihdr['sync_time'],
                                             this_gulp_time,
                                             bw_hz,
                                             sfreq,
                                             upstream_acc_len,
                                             self.nvis_per_packet,
                                             nchan,
                                             chan0,
                                             ) + baselines_flat[vn*4*self.nvis_per_packet : (vn+1)*4*self.nvis_per_packet].tobytes()
                        #print(baselines_flat[vn*4*self.nvis_per_packet : (vn+1)*4*self.nvis_per_packet])
                        #print(dout[vn*self.nvis_per_packet : (vn+1)*self.nvis_per_packet])
                        self.sock.sendto(header + dout[vn*self.nvis_per_packet : (vn+1)*self.nvis_per_packet].tobytes(), (self.dest_ip, self.dest_port))
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.stats.update({'last_end_sample': this_gulp_time})
                self.update_stats()
                # And, update overall time counter
                this_gulp_time += upstream_acc_len
