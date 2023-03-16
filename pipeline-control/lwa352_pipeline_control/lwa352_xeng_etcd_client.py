import os
import sys
import time
import datetime
import socket
import json
import threading
import numpy as np
import etcd3
import logging
import netifaces
import subprocess

PIPELINE_COMMAND = "lwa352-pipeline.py" # used for 'killall'
# DEFAULT PIPELINE SETTINGS
NCHAN = 96
IFACE = ['enp24s0', 'enp216s0', 'enp24s0', 'enp216s0']
RXPORT = [10000, 10000, 20000, 20000]
GPU = [0, 1, 0, 1]
BUFGBYTES = [0, 0, 16, 16] # Only second pipeline per NUMA-node gets a buffer
ETCDHOST = 'etcdv3service.sas.pvt'
CORES = [[1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
         [11,12,13,14,14,14,14,14,14,14,14,14,14,14,14],
         [6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
         [16,17,18,19,19,19,19,19,19,19,19,19,19,19,19]]
CPUMASK = [0x1e, 0x7800, 0x3c0, 0xf0000]
LOGFILE_BASE = os.path.expanduser("~/xpipeline")
PIDFILE_BASE = os.path.expanduser("~/xpipeline")

# ETCD Keys
ETCD_CMD_ROOT = "/cmd/corr/x"
ETCD_MON_ROOT = "/mon/corr/x"
ETCD_RESP_ROOT = "/resp/corr/x"

class LwaXengineEtcdClient():
    """
    An ETCD client to interface a single X-Engine server
    to an etcd store.

    :param etcdhost: Hostname (or IP, in dotted quad notation)
        of target etcd server.
    :type etcdhost: string

    :param etcdport: Port on which etcd is served
    :type etcdport: int

    :param logger: Python `logging.Logger` instance to which
        this class's log messages should be emitted. If None,
        log to stderr
    :type logger: logging.Logger

    """

    def __init__(self, etcdhost="etcdv3service.sas.pvt", etcdport=2379, logger=None):
        #: Hostname of the server running this client
        self.xhost = socket.gethostname()
        #: List of etcd watch IDs, used to kill watch processes
        self._etcd_watch_ids = []
        if logger is None:
            self.logger = logging.getLogger("LwaXengineEtcdClient:%s" % self.xhost)
            stderr_handler = logging.StreamHandler(sys.stderr)
            self.logger.addHandler(stderr_handler)
            self.set_log_level("info")
        else:
            self.logger = logger

        self.ec = etcd3.Etcd3Client(host=etcdhost, port=etcdport)
        try:
            val, meta = self.ec.get('foo')
        except:
            self.logger.exception("Failed to connect to Etcd server on host %s" % etcdhost)
            raise

        self.cmd_key = ETCD_CMD_ROOT + "/%s/xctrl" % (self.xhost)
        self.cmd_resp_key = ETCD_RESP_ROOT + "/%s/xctrl" % (self.xhost)
        self.mon_key = ETCD_MON_ROOT + "/%s/xctrl" % (self.xhost)
        self.logger.debug("Command key is %s" % self.cmd_key)
        self.logger.debug("Command response key is %s" % self.cmd_resp_key)
        self.logger.debug("Monitor key root is %s" % self.mon_key)

        self.xctrl = XengineController(logger=self.logger)

    def set_log_level(self, level):
        """
        Set the logging level.

        :param level: Logging level. Should be "debug", "info", or "warning"
        :type level: string
        """
        if level not in ["debug", "info", "warning"]:
            self.logger.error("Can't set log level to %s. Should be "
                "'debug', 'info', or 'warning'")
            return
        if level == "debug":
            self.logger.setLevel(logging.DEBUG)
        elif level == "info":
            self.logger.setLevel(logging.INFO)
        elif level == "warning":
            self.logger.setLevel(logging.WARNING)

    def start_command_watch(self):
        """
        Listen for commands on this F-Engine's command key, as well as the
        "all SNAPs" key.
        Stop listening with `stop_command_watch()`

        Internally, this method sets the `_etcd_watch_ids` attribute to
        allow a watch to later be cancelled.
        """
        self.logger.info("Beginning command watch on key %s" % self.cmd_key)
        self._etcd_watch_ids += [self.ec.add_watch_prefix_callback(
                                  self.cmd_key,
                                  self._etcd_callback,
                                 )]

    def stop_command_watch(self):
        """
        Stop listening for commands on this F-Engine's command key, as well
        as the "all SNAPs" key.

        Internally, this method sets the `_etcd_watch_ids` attribute to `[]`.
        """
        if self._etcd_watch_ids == []:
            self.logger.info("Trying to stop a non-existent command watch")
        else:
            self.logger.info("Stopping command watch")
            for watch_id in self._etcd_watch_ids:
                self.ec.cancel_watch(watch_id)
            self._etcd_watch_ids = []

    def _send_command_response(self, seq_id, processed_ok, response):
        """
        Respond to a received command with sequence ID `seq_id` on the
        command response etcd channel.

        :param seq_id: Sequence ID of the command to which we are responding.
        :type seq_id: string

        :param processed_ok: Flag indicating the response is an error if False.
        :type status: bool

        :param response: String response with which to respond. E.g.
            'out of range', or 'command accepted'. If the command returns data,
            this might be a json string of this data.
        :type response: string
        """
        if processed_ok:
            status = 'normal'
        else:
            status = 'error'
        resp = {
            'id': seq_id,
            'val': {
                'status': status,
                'response': response,
                'timestamp': time.time(),
            }
        }
        resp_json = json.dumps(resp)
        try:
            self.ec.put(self.cmd_resp_key, resp_json)
        except:
            self.logger.error("Error trying to send ETCD command response")
            raise

    def _etcd_callback(self, watchresponse):
        """
        A callback executed whenever this block's command key is modified.
        
        This callback JSON decodes the key contents, and passes the
        resulting dictionary to ``_process_commands``.

        :param watchresponse: A WatchResponse object used by the etcd
            `add_watch_prefix_callback` as the calling argument.
        :type watchresponse: WatchResponse

        :return: True if command was processed successfully, False otherwise.
        """
        for event in watchresponse.events:
            self.logger.debug("Got command: %s" % event.value)
            try:
                command_dict = json.loads(event.value.decode())
            except json.JSONDecodeError:
                err = "JSON decode error"
                self.logger.error(err)
                # If decode fails, we don't even have a command ID, so send
                # an error with seq_id "Unknown"
                self._send_command_response("Unknown", False, err)
                return False
            self.logger.debug("Decoded command: %s" % command_dict)

            for field in ["id", "cmd", "val"]:
                if not field in command_dict:
                    err = "No '%s' field in message" % field
                    self.logger.error(err)
                    self._send_command_response("Unknown", False, err)
                    return False

            seq_id = command_dict.get("id", "Unknown")
            if not isinstance(seq_id, str):
                err = "Sequence ID not string"
                self.logger.error(err)
                self._send_command_response("Unknown", False, err)
                return False

            try:
                block = command_dict["val"].get("block", None)
            except:
                block = None
            if block is None:
                self.logger.error("Received val string with no 'block' key!")
                err = "Bad command format"
                self._send_command_response(seq_id, False, err)
                return False

            command = command_dict.get("cmd", None)
            if command is None:
                self.logger.error("Received command string with no 'command' key!")
                err = "Bad command format"
                self._send_command_response(seq_id, False, err)
                return False

            # Only allow commands to reference blocks which are in the
            # Fengine.blocks dict, or Fengine itself
            if block == "xctrl":
                block_obj = self.xctrl
            else:
                self.logger.error("Received block %s not allowed!" % block)
                err = "Wrong block"
                self._send_command_response(seq_id, False, err)
                return False

            # Check command is valid
            if command.startswith("_"):
                self.logger.error("Received command starting with underscore!")
                err = "Command not allowed"
                self._send_command_response(seq_id, False, err)
                return False
            if not (hasattr(block_obj, command) and callable(getattr(block_obj, command))):
                self.logger.error("Received command invalid!")
                err = "Command invalid"
                self._send_command_response(seq_id, False, err)
                return False
            else:
                cmd_method = getattr(block_obj, command)
            # Process command
            cmd_kwargs = command_dict["val"].get("kwargs", {})
            ok = True
            try:
                #if self.is_polling():
                #    self._poll_pause_trigger.set()
                #    self._poll_is_paused.wait(timeout=10)
                resp = cmd_method(**cmd_kwargs)
                #self._poll_pause_trigger.clear()
            except TypeError:
                ok = False
                err = "Command arguments invalid"
                self.logger.exception(err)
            except:
                ok = False
                err = "Command failed"
                self.logger.exception(err)
            if not ok:
                self._send_command_response(seq_id, ok, err)
                self.logger.error("Responded to command '%s' (ID %s): %s" % (command, seq_id, err))
                return False
            try:
                if isinstance(resp, np.ndarray):
                    resp = resp.tolist()
                # Check we will be able to encode the response
                test_encode = json.dumps(resp)
            except:
                self.logger.exception("Failed to encode JSON")
                resp = "JSON_ERROR"
            self._send_command_response(seq_id, ok, resp)
            self.logger.info("Responded to command '%s' (ID %s): OK? %s" % (command, seq_id, ok))
            self.logger.debug("Responded to command '%s' (ID %s): %s" % (command, seq_id, resp))
            return ok

    def __del__(self):
        self.stop_command_watch()

class XengineController():
    def __init__(self, logger=None):
        self.hostname = socket.gethostname()
        if logger is None:
            self.logger = logging.getLogger("XengineController:%s" % (self.hostname))
            stderr_handler = logging.StreamHandler(sys.stderr)
            self.logger.addHandler(stderr_handler)
            self.set_log_level("info")
        else:
            self.logger = logger

    def _pidfile(self, xid):
        return "%s.%d.pid" % (PIDFILE_BASE, xid)

    def set_log_level(self, level):
        """
        Set the logging level.

        :param level: Logging level. Should be "debug", "info", or "warning"
        :type level: string
        """
        if level not in ["debug", "info", "warning"]:
            self.logger.error("Can't set log level to %s. Should be "
                "'debug', 'info', or 'warning'")
            return
        if level == "debug":
            self.logger.setLevel(logging.DEBUG)
        elif level == "info":
            self.logger.setLevel(logging.INFO)
        elif level == "warning":
            self.logger.setLevel(logging.WARNING)

    def get_pid(self, xid):
        p = self._pidfile(xid)
        if not os.path.isfile(p):
            self.logger.error("No PIDfile (%s) available" % p)
            return None
        with open(p, 'r') as fh:
            pid = int(fh.read())
        return pid

    def set_pid(self, xid, pid):
        p = self._pidfile(xid)
        if pid is None:
            self.logger.info("Removing pidfile %s" % p)
            subprocess.run(["rm", p])
        else:
            self.logger.info("Setting pidfile %s to %d" % (p, pid))
            with open(p, 'w') as fh:
                fh.write(str(pid))

    def stop_pipeline(self, xid, force=False):
        pid = self.get_pid(xid)
        if not force and pid is not None:
            self.logger.info("Killing process %d" % pid)
            subprocess.run(["kill", "-9", str(pid)])
        if force:
            subprocess.run(["killall", "-9", PIPELINE_COMMAND])
        self.set_pid(xid, None)

    def start_pipeline(self,
          xid,
          cpumask=None,
          nchan=NCHAN,
          gpudev=None,
          etcdhost=ETCDHOST,
          interface=None,
          rxport=None,
          bufgbytes=None,
          cores=None,
          logfile=None,
        ):

        if cpumask is None:
            cpumask = CPUMASK[xid]
        if gpudev is None:
            gpudev = GPU[xid]
        if interface is None:
            interface = IFACE[xid]
        if rxport is None:
            rxport = RXPORT[xid]
        if cores is None:
            cores = CORES[xid]
        if bufgbytes is None:
            bufgbytes = BUFGBYTES[xid]
        rxip = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
        logfile = logfile or "%s.%s.%d.log" % (LOGFILE_BASE, self.hostname, xid)
        cmd = [
                  '/usr/bin/taskset', '0x%x' % cpumask,
                  PIPELINE_COMMAND,
                  '--nchan', str(nchan),
                  '--ibverbs',
                  '--gpu', str(gpudev),
                  '--pipelineid', str(xid),
                  '--useetcd',
                  '--etcdhost', etcdhost,
                  '--ip', rxip,
                  '--port', str(rxport),
                  '--bufgbytes', str(bufgbytes),
                  '--cores', ','.join(map(str, cores)),
                  '--logfile', logfile,
              ]
        self.logger.info('Running:' + ' '.join(cmd))
        process = subprocess.Popen(cmd)
        self.set_pid(xid, process.pid)
        
