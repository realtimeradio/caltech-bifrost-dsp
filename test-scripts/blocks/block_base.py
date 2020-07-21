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

from threading import Lock

class Block(object):
    """
    The base class for a bifrost LWA352 processing block
    """
    pipeline_id = 0
    def __init__(self, log, iring, oring,
            guarantee, core, etcd_client=None,
            command_keyroot='/cmd/corr',
            monitor_keyroot='/mon/corr'):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.guarantee = guarantee
        self.core = core

        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})

        # optional etcd client
        self.etcd_client = etcd_client
        self.command_key = command_keyroot + '/' + socket.gethostname() + '/' + str(self.pipeline_id)
        self.monitor_key = monitor_keyroot + '/' + socket.gethostname() + '/' + str(self.pipeline_id)

        self.etcd_watch_id = None

        self.control_lock = Lock()

    def acquire_control_lock(self):
        self.control_lock.acquire()

    def release_control_lock(self):
        self.control_lock.release()

    def __del__(self):
        if self.etcd_watch_id:
            self.etcd_client.cancel_watch(self.etcd_watch_id)
