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
    _instance_count = -1

    @classmethod
    def set_id(cls, x):
       cls.pipeline_id = x
    _count = -1

    @classmethod
    def _get_instance_id(cls):
        """
        Get an auto-incrementing ID number for a Block of a particular
        class type.
        :param cls: ``Block`` class, e.g. ``BeamformOutputBlock``
        :return: Number of instances of this class currently constructed.
        """
        cls._instance_count += 1
        return cls._instance_count

    def __init__(self, log, iring, oring,
            guarantee, core, etcd_client=None,
            command_keyroot='/cmd/corr',
            monitor_keyroot='/mon/corr',
            name=None):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.guarantee = guarantee
        self.core = core
        self.instance_id = self._get_instance_id()
        self.name = name or type(self).__name__
        self.stats = {}

        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")

        self.stats_proclog = ProcLog(type(self).__name__+"/stats")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        if self.oring is not None:
            self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})

        # optional etcd client
        self.etcd_client = etcd_client
        self.command_key = '{cmdroot}/x/{host}/pipeline/{pid}/{block}/{id}/ctrl'.format(
                                cmdroot=command_keyroot,
                                host=socket.gethostname(),
                                pid=self.pipeline_id,
                                block=self.name,
                                id=self.instance_id)
        self.monitor_key = '{monroot}/x/{host}/pipeline/{pid}/{block}/{id}'.format(
                                monroot=monitor_keyroot,
                                host=socket.gethostname(),
                                pid=self.pipeline_id,
                                block=self.name,
                                id=self.instance_id)

        self.etcd_watch_id = None

        # A lock to protect actions occuring in the etcd callback
        self.control_lock = Lock()

        if self.etcd_client:
            self.log.info("Adding watch callback to %s" % self.command_key)
            self.etcd_watch_id = self.etcd_client.add_watch_prefix_callback(self.command_key, self._etcd_callback)

    def _etcd_callback(self, watchresponse):
        """
        A callback executed whenever this block's command key is written to.
        parameters:
            watchresponse: A WatchResponse object used by the etcd `add_watch_prefix_callback` as the calling argument.
        """
        pass

    def acquire_control_lock(self):
        self.control_lock.acquire()

    def release_control_lock(self):
        self.control_lock.release()

    def update_stats(self):
        self.stats_proclog.update(self.stats)

    def __del__(self):
        if self.etcd_watch_id:
            self.etcd_client.cancel_watch(self.etcd_watch_id)
