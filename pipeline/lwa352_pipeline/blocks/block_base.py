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
import ujson as json
import socket

from threading import Lock

COMMAND_OK = 0
COMMAND_NOT_RECOGNIZED = -1
COMMAND_WRONG_TYPE = -2
COMMAND_INVALID = -3

class Block(object):
    """
    The base class for a bifrost LWA352 processing block

    **Instantiation**

    :param log: Logging object to which runtime messages should be
        emitted.
    :type log: logging.Logger

    :param iring: bifrost input data ring. This should be on the GPU.
    :type iring: bifrost.ring.Ring

    :param oring: bifrost output data ring. This should be on the GPU.
    :type oring: bifrost.ring.Ring

    :param guarantee: If True, read data from the input ring in blocking "guaranteed"
        mode, applying backpressure if necessary to ensure no data are missed by this block.
    :type guarantee: Bool

    :param core: CPU core to which this block should be bound. A value of -1 indicates no binding.
    :type core: int

    :param gpu: GPU device which this block should target. A value of -1 indicates no binding
    :type gpu: int

    :param ntime_gulp: Number of time samples to copy with each gulp.
    :type ntime_gulp: int

    :parameter etcd_client: Etcd client object used to facilitate control of this block.
        If ``None``, do not use runtime control.
    :type etcd_client: etcd3.client.Etcd3Client

    :parameter name: Custom name for this block for use in generating control
        and monitoring keys. If None, the class name is used.
    :type name: string

    :parameter command_keyroot: Root path under which all command keys live.
        The complete path is ``command_keyroot``/x/``hostname``/pipeline/``pipeline_id``/``block_name``/``block_id``/ctrl.
        Where fields are:
          -  ``hostname`` : The network hostname of the server running the pipeline
             as returned by ``socket.hostname()``
          -  ``pipeline_id`` : The pipeline ID, as set by the user using the
             class method ``Block.set_id()`` prior to instantion,
          -  ``block_name`` : The block name as passed using the ``name`` parameter, or
             as inferred from the block class name
          - ``block_id`` : A zero indexed counter unique to each subclass, which
            increments with each instantiation. I.e., if 3 ``BeamformBlock`` instances
            are created, these will be given block IDs 0, 1, and 2.
    :type command_keyroot: string

    :parameter monitor_keyroot: Root path under which all monitor keys live.
        The complete path is  ``monitort_keyroot``/x/``hostname``/pipeline/``pipeline_id``/``block_name``/``block_id``/status.
        Where fields are as defined for the ``command_root`` parameter.
    """
    pipeline_id = 0
    _instance_count = -1

    @classmethod
    def set_id(cls, x):
       cls.pipeline_id = x

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

        self.log.info("Pipeline %d: Initializing block: %s (instance %d)" % (self.pipeline_id, self.name, self.instance_id))

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
        self.monitor_key = '{monroot}/x/{host}/pipeline/{pid}/{block}/{id}/status'.format(
                                monroot=monitor_keyroot,
                                host=socket.gethostname(),
                                pid=self.pipeline_id,
                                block=self.name,
                                id=self.instance_id)

        self._etcd_watch_id = None

        # A lock to protect actions occuring in the etcd callback
        self._control_lock = Lock()

        if self.etcd_client:
            self.log.info("Adding watch callback to %s" % self.command_key)
            self._etcd_watch_id = self.etcd_client.add_watch_prefix_callback(self.command_key, self._etcd_callback)

        self.update_pending = False
        self.command_vals = {}
        self._pending_command_vals = {}
        self._command_types = {}
        self._command_conditions = {}

    def define_command_key(self, name, type=None, condition=None, initial_val=None): 
        """
        Add (or redefine) a new command key which this block uses.

        :param name: Name of the key
        :type name: str

        :param type: Type of the value associated with this key,
            eg. int/str/list. If the received command value fails the test
            ``isinstance(<value>, type)`` the command will be rejected

        :param condition: A [lambda] function to be called with the value
            held by the received command key as its argument. If the
            function returns False, the key will be rejected. Eg., to ensure
            a value associated with a key always has an even value, pass
            ``condition = lambda x: x%2 == 0``
        :type condition: function

        :param initial_val: An initial value for the command key to hold
        :type initial_val: ``type``
        """
        
        if initial_val:
            if type:
                assert isinstance(initial_val, type), "%s: key %s: Initial value type check fail!" % (self.name, name)
            if condition:
                assert condition(initial_val), "%s: key %s: Intial value failed condition check! (%s)" % (self.name, name, condition)
        self.command_vals[name] = initial_val
        self._pending_command_vals[name] = initial_val
        self._command_types[name] = type
        self._command_conditions[name] = condition

    def _etcd_callback(self, watchresponse):
        """
        A callback executed whenever this block's command key is modified.

        This callback JSON decodes the key contents, and passes the
        resulting dictionary to ``_process_commands``.
        The ``last_cmd_response`` status value is set to the return value of
        ``_process_commands`` to indicate any error conditions

        :param watchresponse: A WatchResponse object used by the etcd
            `add_watch_prefix_callback` as the calling argument.
        :type watchresponse: WatchResponse
        """
        cpu_affinity.set_core(self.core)
        self._control_lock.acquire()
        for event in watchresponse.events:
            v = json.loads(event.value)
            self.update_stats({'last_cmd_response':self._process_commands(v)})
        self._control_lock.release()

    def _process_commands(self, command_dict):
        """
        Take a dictionary of commands and load them into the ``_pending_command_vals``
        dictionary after checking data types and other user-provided
        conditions. After loading all command key/value pairs, set the
        ``update_pending`` attribute of this block to True, to indicate
        that updated command values are available. Also set the ``last_cmd_time``
        status key with ``time.time()`` to indicate that a command set
        was successfully parsed.

        This method will return immediately if a command value fails a check,
        leaving an undefined number of pending command keys set. If this happens,
        the block's ``update_pending`` and ``last_cmd_time`` attributes are not set

        :param command_dict: Dictionary of command keys and their values, eg.
            {'dest_ip': '100.100.10.1', 'dest_port': 10000}
        :type command_dict: dict

        :return: Return COMMAND_OK (0) if everything completes successfully.
            Return COMMAND_NOT_RECOGNIZED (-1) if a command key is not known to
            this block; COMMAND_WRONG_TYPE (-2) if a command has a value of
            the wrong type; COMMAND_INVALID (-3) if a command has a value which
            fails a custom user test.
        :rtype: int
        """

        cpu_affinity.set_core(self.core)
        for key in command_dict.keys():
            if key not in self.command_vals.keys():
                self.log.error("%s: Command key %s not recognized" % (self.name, key))
                return COMMAND_NOT_RECOGNIZED
            if self._command_types[key]:
                if not isinstance(command_dict[key], self._command_types[key]):
                    self.log.error("%s: Command key %s had wrong type (had %s, expected %s)" %
                                   (self.name, key, type(command_dict[key]), self._command_types[key]))
                    return COMMAND_WRONG_TYPE
            if self._command_conditions[key]:
                if not self._command_conditions[key](command_dict[key]):
                    self.log.error("%s: Command key %s failed requirements" % (self.name, key))
                    return COMMAND_INVALID
            self._pending_command_vals[key] = command_dict[key]
            # Track the command status in the stats log
            self.stats['new_' + key] = command_dict[key]
        self.update_pending = True
        self.stats['update_pending'] = True
        self.stats['last_cmd_time'] = time.time()
        return COMMAND_OK

    def update_command_vals(self):
        """
        Copy command entries from the ``_pending_command_vals``
        dictionary to the ``command_vals`` dictionary, to be used
        by the block's runtime processing.
        Set the ``update_pending`` flag to False, to indicate that 
        there are no longer waiting commands. Set the status key
        ``last_cmd_proc_time`` to ``time.time()`` to record the
        time at which this method was called.
        """
        cpu_affinity.set_core(self.core)
        self._control_lock.acquire()
        self.command_vals.update(self._pending_command_vals)
        self.update_pending = False
        self.stats['update_pending'] = False
        self.stats['last_cmd_proc_time'] = time.time()
        self._control_lock.release()
        self.update_stats(self.command_vals)

    def acquire_control_lock(self):
        self._control_lock.acquire()

    def release_control_lock(self):
        self._control_lock.release()

    def update_stats(self, new_stats={}):
        """
        Update the stats proclog. This is a wrapper
        around stats_proclog.update, which *replaces*
        the contents of the log. This method
        augments the contents.

        :param new_stats: Status values to create or update
        :type new_stats: dict
        """
        # This function updates without deleting everything
        self.stats.update(new_stats)
        # This function deletes everything and replaces
        self.stats_proclog.update(self.stats)

    def __del__(self):
        if self._etcd_watch_id:
            self.etcd_client.cancel_watch(self._etcd_watch_id)
