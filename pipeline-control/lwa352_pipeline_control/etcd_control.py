import argparse
import time
import sys
import logging
import simplejson as json
import etcd3 as etcd

default_log = logging.getLogger(__name__)
logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
logFormat.converter = time.gmtime
logHandler = logging.StreamHandler(sys.stdout)
logHandler.setFormatter(logFormat)
default_log.addHandler(logHandler)

class EtcdCorrControl():
    """
    **Description**

    A class to encapsulate the LWA correlator etcd-based control protocol,
    whereby commands are sent to processing blocks in the LWA pipeline
    by writing json-encoded strings to an appropriate etcd key.

    The target key for a particular processing block is defined as:

    ``<keyroot_cmd>/<hostname>/pipeline/<pipeline-id>/<blockname>/<block-id>``

    Where:
      -  ``command-root`` is a user-defined root path
      -  ``hostname`` is the hostname of the server running the DSP pipeline
      -  ``pipeline-id`` is the index of the DSP pipeline on this server
      -  ``blockname`` is the name of the processing block in the pipeline
      -  ``block-id`` is the index of the instance of this block type

    A block will respond to commands on a similarly defined key:
    ``<keyroot_resp>/<hostname>/pipeline/<pipeline-id>/<blockname>/<block-id>``

    Similarly, a processing block's status can be read by reading the key:

    ``<keyroot_mon>/<hostname>/pipeline/<pipeline-id>/<blockname>/<block-id>``

    Where:
      -  ``monitor-root`` is a user-defined root path
      -  Other key elements are defined as per the command key above

    This class facilitates the generation of appropriate keys given a
    target host/pipeline/block, and handles interactions with the correlator's
    etcd server as well as the encoding and decoding of JSON messages.

    **Instantiation**

    :param etcdhost: The hostname of the system running the correlator's
        etcd server
    :type etcdhost: string

    :param keyroot_cmd: The root path under which all correlator command
        keys live
    :type keyroot_cmd: string

    :param keyroot_mon: The root path under which all correlator command
        response keys live
    :type keyroot_mon: string

    :param keyroot_mon: The root path under which all correlator monitor
        keys live
    :type keyroot_mon: string

    :param log: The logger to which this class should emit log messages.
        The default behaviour is to log to stdout
    :type log: logging.Logger

    :param simulated: If True, don't send messages over etcd, just
        return their JSON strings.
    :type simulated: bool

    """
    def __init__(self, etcdhost='etcdhost', keyroot_cmd='/cmd/corr/x',
                 keyroot_mon='/mon/corr/x', keyroot_resp='/resp/corr/x',
                 log=default_log, simulated=False):
        self.keyroot_cmd = keyroot_cmd
        self.keyroot_mon = keyroot_mon
        self.keyroot_resp = keyroot_resp
        self.etcdhost = etcdhost
        self.log = log
        self.simulated = simulated
        if simulated:
            self.ec = None
        else:
            try:
                self.ec = etcd.client(self.etcdhost)
            except:
                log.error('Failed to connect to ETCD host %s' % self.etcdhost)
                raise

    def _get_cmd_key(self, host, pipeline, block, inst_id):
        """
        Generate a block's command key from the block instance specification.

        :param host: The hostname of the server running the DSP pipeline
            to be commanded
        :type host: string
        :param pipeline: The index of the pipeline on this server to be
            commanded
        :type pipeline: int
        :param block: The name of the processing block in this pipeline
            to be commanded
        :type block: string
        :param inst_id: The instance ID of the block of this type to be
            commanded
        :type inst_id: int

        :return: The command key for this block
        :rtype: string

        """

        key = self._get_key(host, pipeline, block, inst_id)
        return self.keyroot_cmd + key

    def _get_resp_key(self, host, pipeline, block, inst_id):
        """
        Generate a block's response key from the block instance specification.

        :param host: The hostname of the server running the DSP pipeline
            to be commanded
        :type host: string
        :param pipeline: The index of the pipeline on this server to be
            commanded
        :type pipeline: int
        :param block: The name of the processing block in this pipeline
            to be commanded
        :type block: string
        :param inst_id: The instance ID of the block of this type to be
            commanded
        :type inst_id: int

        :return: The response key for this block
        :rtype: string

        """

        key = self._get_key(host, pipeline, block, inst_id)
        return self.keyroot_resp + key

    def _get_key(self, host, pipeline, block, inst_id):
        """
        Generate a block's key suffix from the block instance specification.

        :param host: The hostname of the server running the DSP pipeline
            to be commanded
        :type host: string
        :param pipeline: The index of the pipeline on this server to be
            commanded
        :type pipeline: int
        :param block: The name of the processing block in this pipeline
            to be commanded
        :type block: string
        :param inst_id: The instance ID of the block of this type to be
            commanded
        :type inst_id: int

        :return: The monitor key for this block
        :rtype: string

        """
        key = '/%s' % (host)
        if pipeline is not None:
            key += '/pipeline/%d' % (pipeline)
        if block is not None:
            key += '/%s' % (block)
        if inst_id is not None:
            key += '/%d' % (inst_id)
        return key

    def _get_mon_key(self, host, pipeline, block, inst_id):
        """
        Generate a block's monitor key from the block instance specification.

        :param host: The hostname of the server running the DSP pipeline
            to be commanded
        :type host: string
        :param pipeline: The index of the pipeline on this server to be
            commanded
        :type pipeline: int
        :param block: The name of the processing block in this pipeline
            to be commanded
        :type block: string
        :param inst_id: The instance ID of the block of this type to be
            commanded
        :type inst_id: int

        :return: The monitor key for this block
        :rtype: string

        """
        key = self._get_key(host, pipeline, block, inst_id)
        return self.keyroot_mon + key

    def send_command(self, host, pipeline=None, block=None, inst_id=None,
            cmd='update', timeout=10.0, **kwargs):
        """
        Send a command to a processing block

        :param host: The hostname of the server running the DSP pipeline
            to be commanded
        :type host: string
        :param pipeline: The index of the pipeline on this server to be
            commanded. Use None for commands targetting a raw host.
        :type pipeline: int
        :param block: The name of the processing block in this pipeline
            to be commanded. Use None for commands targetting a raw host
        :type block: string
        :param inst_id: The instance ID of the block of this type to be
            commanded. Use None for commands targetting a raw host
        :type inst_id: int
        :param cmd: Command name
        :type cmd: str
        :param timeout: Time, in seconds, to wait for a response to the command.
        :type timeout: float

        :param **kwargs: Keyword arguments are used to specify which
            control values should be set. Any key names and JSON-serializable
            values are allowed. These should match the key names expected
            by the processing block being targeted.
        :type **kwargs: Any JSON-serializable values

        If ``self.simulated=True``, returns the JSON string which would be sent over
        etcd.

        """

        cmd_key = self._get_cmd_key(host, pipeline, block, inst_id)
        resp_key = self._get_resp_key(host, pipeline, block, inst_id)
        timestamp = time.time()
        sequence_id = str(int(timestamp * 1e6))
        command_json = self._format_command(
                           sequence_id,
                           timestamp,
                           block,
                           cmd,
                           kwargs = kwargs,
                       )
        if command_json is None:
            return False

        if self.simulated:
            return command_json

        self._response_received = False
        self._response = None

        def response_callback(watchresponse):
            for event in watchresponse.events:
                self.log.debug("Got command response")
                try:
                    response_dict = json.loads(event.value.decode())
                except:
                    self.log.exception("Response JSON decode error")
                    continue
                self.log.debug("Response: %s" % response_dict)
                resp_id = response_dict.get("id", None)
                if resp_id == sequence_id:
                    self._response = response_dict
                    self._response_received = True
                else:
                    self.log.debug("Seq ID %s didn't match expected (%s)" % (resp_id, sequence_id))

        # Begin watching response channel and then send message
        watch_id = self.ec.add_watch_callback(resp_key, response_callback)
        # send command
        print(cmd_key, resp_key, command_json)
        self.ec.put(cmd_key, command_json)
        starttime = time.time()
        while(True):
            if self._response_received:
                self.ec.cancel_watch(watch_id)
                status = self._response['val']['status']
                if status != 'normal':
                    self.log.info("Command status returned: '%s'" % status)
                return self._response['val']['response']
            if time.time() > starttime + timeout:
                self.ec.cancel_watch(watch_id)
                self.log.error("host %s (pipeline %s) failed to respond to etcd command!" % (host, str(pipeline)))
                raise RuntimeError
            time.sleep(0.001)

    def _format_command(self, sequence_id, timestamp, block, cmd, kwargs={}):
        """
        Format a command to be sent via ETCD

        :param sequence_id: The ``id`` command field
        :type block: int

        :param timestamp: The ``timestamp`` command field
        :type timestamp: float

        :param block: The ``block`` command field
        :type block: str

        :param cmd: The ``cmd`` command field
        :type cmd: str

        :param kwargs: The ``kwargs`` command field
        :type kwargs: dict

        :return: JSON-encoded command string to be sent. Returns None if there
            is an enoding error.
        """
        command_dict = {
            "cmd": cmd,
            "val": {
                "block": block,
                "timestamp": timestamp,
                "kwargs": kwargs,
                },
            "id": sequence_id,
        }
        try:
            command_json = json.dumps(command_dict)
            return command_json
        except:
            self.log.exception("Failed to JSON-encode command")
            return

    def get_status(self, host, pipeline, block, inst_id, user_only=True):
        """
        Read a processing blocks status dictionary

        :param host: The hostname of the server running the DSP pipeline
            to be commanded
        :type host: string
        :param pipeline: The index of the pipeline on this server to be
            commanded
        :type pipeline: int
        :param block: The name of the processing block in this pipeline
            to be commanded
        :type block: string
        :param inst_id: The instance ID of the block of this type to be
            commanded
        :type inst_id: int
        :param user_only: If ``True`` read only statistics which are
            part of the block's user-level reporting capability. Otherwise,
            read all statistics, including those provided by the bifrost
            framework. In this case, user-level statistics are returned
            under the key name "stats"
        :type user_only: Bool

        :return: A dictionary of status values

        """
        if self.simulated:
            self.log.error("Can't get status in simulation mode")
            return {}

        key = self._get_mon_key(host, pipeline, block, inst_id)
        val, meta = self.ec.get(key)
        if val is None:
            self.log.warning("Etcd key %s returned no data" % key)
            return val
        val = json.loads(val)
        if user_only:
            return val.get("stats", {})
        else:
            return val


