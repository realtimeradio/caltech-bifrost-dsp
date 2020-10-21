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

class BeamformVlbiOutput(Block):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring,
                 guarantee=True, core=-1, etcd_client=None, dest_port=10000,
                 ntime_gulp=480,
                 ):
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        super(BeamformVlbiOutput, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        cpu_affinity.set_core(self.core)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest_ip = '0.0.0.0'
        self.new_dest_ip = '0.0.0.0'
        self.dest_port = dest_port
        self.new_dest_port = dest_port
        self.packet_delay_ns = 1
        self.new_packet_delay_ns = 1
        self.update_pending = True
        self.ntime_gulp = ntime_gulp

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
        if 'packet_delay_ns' in v:
            self.new_packet_delay_ns = v['packet_delay_ns']
        self.update_pending = True
        self.stats.update({'new_dest_ip': self.new_dest_ip,
                           'new_dest_port': self.new_dest_port,
                           'new_packet_delay_ns': self.new_packet_delay_ns,
                           'update_pending': self.update_pending,
                           'last_cmd_time': time.time()})
        self.update_stats()

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        prev_time = time.time()
        for iseq in self.iring.read(guarantee=self.guarantee):
            # Update control each sequence
            self.update_pending = True
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            nchan = ihdr['nchan']
            nbeam = ihdr['nbeam']
            nbit  = ihdr['nbit']
            nchan = ihdr['nchan']
            chan0 = ihdr['chan0']
            bw_hz = ihdr['bw_hz']
            sfreq = ihdr['sfreq']
            npol  = ihdr['npol']
            igulp_size = self.ntime_gulp * nbeam * nchan * npol * nbit // 8
            idata_cpu = BFArray(shape=[self.ntime_gulp, nchan, 2, 2], dtype='f32', space='cuda_host')
            packet_cnt = 0
            for ispan in iseq.read(igulp_size):
                # Update destinations if necessary
                if self.update_pending:
                    self.dest_ip = self.new_dest_ip
                    self.dest_port = self.new_dest_port
                    self.packet_delay_ns = self.new_packet_delay_ns
                    self.update_pending = False
                    self.log.info("VLBI OUTPUT >> Updating destination to %s:%s (packet delay %d ns)" % (self.dest_ip, self.dest_port, self.packet_delay_ns))
                    self.stats.update({'dest_ip': self.dest_ip,
                                       'dest_port': self.dest_port,
                                       'packet_delay_ns': self.packet_delay_ns,
                                       'update_pending': self.update_pending,
                                       'last_update_time': time.time()})
                self.stats['curr_sample'] = this_gulp_time
                self.update_stats()
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                if self.dest_ip != '0.0.0.0':
                    idata = ispan.data.view('f32').reshape([self.ntime_gulp, nchan, nbeam, 2])
                    #idata_cpu[:, :, 0, :] = idata[:, :, 0, :]
                    #idata_cpu[:, :, 1, :] = idata[:, :, 1, :]
                    start_time = time.time()
                    header = struct.pack('>QQ2d3I',
                                         ihdr['sync_time'],
                                         this_gulp_time,
                                         bw_hz,
                                         sfreq,
                                         nchan,
                                         chan0,
                                         npol,
                                        )
                    for t in range(self.ntime_gulp):
                        #header = struct.pack('>QQ2d3I',
                        #                     ihdr['sync_time'],
                        #                     this_gulp_time + t,
                        #                     bw_hz,
                        #                     sfreq,
                        #                     nchan,
                        #                     chan0,
                        #                     npol,
                        #                    )
                        self.sock.sendto(header + idata_cpu[t].tobytes(), (self.dest_ip, self.dest_port))
                        packet_cnt += 1
                        if packet_cnt % 50 == 0:
                            # Only implement packet delay every 50 packets because the sleep
                            # overhead is large
                            time.sleep(50 * self.packet_delay_ns / 1e9)
                    stop_time = time.time()
                    elapsed = stop_time - start_time
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.stats['last_end_sample'] = this_gulp_time
                self.update_stats()
                # And, update overall time counter
                this_gulp_time += self.ntime_gulp
