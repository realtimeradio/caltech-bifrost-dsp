import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap
from bifrost.ndarray import copy_array
from bifrost.device import stream_synchronize, set_device as BFSetGPU
from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_writer import HeaderInfo, DiskWriter, UDPTransmit

import os
import time
import simplejson as json
import socket
import struct
import numpy as np

from .block_base import Block

class BeamformVlbiOutput(Block):
    """
    """
    def __init__(self, log, iring,
                 guarantee=True, core=-1, etcd_client=None, dest_port=10000,
                 ntime_gulp=480,
                 ):
        super(BeamformVlbiOutput, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        cpu_affinity.set_core(self.core)

        self.sock = None
        self.dest_ip = '0.0.0.0'
        self.new_dest_ip = '0.0.0.0'
        self.new_dest_ip = '0.0.0.0'
        self.dest_port = dest_port
        self.new_dest_port = dest_port
        self.update_pending = True
        self.ntime_gulp = ntime_gulp

    def _etcd_callback(self, watchresponse):
        """
        A callback to run whenever this block's command key is updated.
        Decode new destination_ip and port
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
            # Update control each sequence
            self.update_pending = True
            ihdr = json.loads(iseq.header.tostring())
            this_gulp_time = ihdr['seq0']
            nchan = ihdr['nchan']
            nbeam = ihdr['nbeam']
            nbit  = ihdr['nbit']
            nchan = ihdr['nchan']
            system_nchan = ihdr['system_nchan']
            chan0 = ihdr['chan0']
            bw_hz = ihdr['bw_hz']
            sfreq = ihdr['sfreq']
            npol  = ihdr['npol']
            igulp_size = self.ntime_gulp * nbeam * nchan * npol * 2 * nbit // 8
            idata_cpu = BFArray(shape=[self.ntime_gulp, nchan, self.nbeam_send], dtype='cf64', space='cuda_host')
            packet_cnt = 0
            udt = None
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # ignore final gulp
                # Update destinations if necessary
                if self.update_pending:
                    self.dest_ip = self.new_dest_ip
                    self.dest_port = self.new_dest_port
                    self.update_pending = False
                    self.log.info("VLBI OUTPUT >> Updating destination to %s:%s (packet delay %d ns)" % (self.dest_ip, self.dest_port, self.max_mbps))
                    if self.sock: del self.sock
                    if udt: del udt
                    self.sock = UDPSocket()
                    self.sock.connect(Address(self.dest_ip, self.dest_port))
                    udt = UDPTransmit('ibeam%i_%i' % (self.nbeam_send, nchan), sock=self.sock, core=self.core)
                    desc = HeaderInfo()
                    desc.set_nchan(system_nchan)
                    desc.set_chan0(chan0)
                    desc.set_nsrc(system_nchan // nchan)
                    desc.set_tuning(0)
                    self.stats.update({'dest_ip': self.dest_ip,
                                       'dest_port': self.dest_port,
                                       'update_pending': self.update_pending,
                                       'last_update_time': time.time()})
                self.stats['curr_sample'] = this_gulp_time
                self.update_stats()
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                if self.dest_ip != '0.0.0.0':
                    start_time = time.time()
                    idata = ispan.data.view('cf64').reshape([self.ntime_gulp, nchan, nbeam])
                    idata_cpu[...] = idata[:, :, 0:self.nbeam_send]
                    idata_cpu = idata_cpu.reshape(self.ntime_gulp, 1, nchan*self.nbeam_send)
                    try:
                        udt.send(desc, this_gulp_time, 1, chan0 // nchan, 1, idata_cpu)
                    except Exception as e:
                        self.log.error("VLBI OUTPUT >> Sending error: %s" % str(e))
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
            del udt
