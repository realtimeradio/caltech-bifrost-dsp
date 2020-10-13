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

class BeamformOutput(Block):
    """
    Perform GPU side accumulation and then transfer to CPU
    """
    def __init__(self, log, iring,
                 guarantee=True, core=-1, etcd_client=None, dest_port=10000, beam_id=0,
                 ):
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        super(BeamformOutput, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.dest_ip = None
        self.new_dest_ip = None
        self.dest_port = dest_port
        self.new_dest_port = dest_port
        self.packet_delay_ns = 1
        self.new_packet_delay_ns = 1
        self.update_pending = True
        self.beam_id = beam_id
        self.igulp_size = 1024

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
            #ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = 0#ihdr['seq0']
            #upstream_acc_len = ihdr['acc_len']
            #upstream_start_time = this_gulp_time
            for ispan in iseq.read(self.igulp_size):
                # Update destinations if necessary
                if self.update_pending:
                    self.dest_ip = self.new_dest_ip
                    self.dest_port = self.new_dest_port
                    self.packet_delay_ns = self.new_packet_delay_ns
                    self.update_pending = False
                    self.log.info("CORR OUTPUT >> Updating destination to %s:%s (packet delay %d ns)" % (self.dest_ip, self.dest_port, self.packet_delay_ns))
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
                #if self.dest_ip is not None:
                #    #dout = np.zeros([self.npol, self.npol, self.nchan, 2], dtype='>i')
                #    #packet_cnt = 0
                #    #for s0 in range(self.nstand):
                #    #    for s1 in range(s0, self.nstand):
                #    #        header = struct.pack(">QQ6L",
                #    #                             ihdr['sync_time'],
                #    #                             this_gulp_time,
                #    #                             upstream_acc_len,
                #    #                             ihdr['chan0'],
                #    #                             self.npol,
                #    #                             self.nchan,
                #    #                             s0, s1)
                #    #        dout = self.reordered_data[s0, s1]
                #    #        self.sock.sendto(header + dout.tobytes(), (self.dest_ip, self.dest_port))
                #    #        packet_cnt += 1
                #    #        if packet_cnt % 10 == 0:
                #    #            # Only implement packet delay every 10 packets because the sleep
                #    #            # overhead is large
                #    #            time.sleep(10 * self.packet_delay_ns / 1e9)
                #    #self.log.info("CORR OUTPUT >> Sending complete for time %d" % this_gulp_time)
                #else:
                #    self.log.info("CORR OUTPUT >> Skipping sending for time %d" % this_gulp_time)
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0, 
                                          'process_time': process_time,})
                self.stats['last_end_sample'] = this_gulp_time
                self.update_stats()
                # And, update overall time counter
                #this_gulp_time += upstream_acc_len
