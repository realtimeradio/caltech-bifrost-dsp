import argparse
import time
import sys
import logging
import socket
import json
import numpy as np

from .etcd_control import EtcdCorrControl
from .lwa352_utils import NCHAN

from .blocks.block_control_base import BlockControl
from .blocks.corr_output_full_control import CorrOutputFullControl
from .blocks.corr_output_part_control import CorrOutputPartControl
from .blocks.corr_acc_control import CorrAccControl
from .blocks.corr_control import CorrControl
from .blocks.corr_subsel_control import CorrSubselControl
from .blocks.triggered_dump_control import TriggeredDumpControl
from .blocks.beamform_control import BeamformControl
from .blocks.beamform_output_control import BeamformOutputControl
from .blocks.beamform_vlbi_output_control import BeamformVlbiOutputControl

default_log = logging.getLogger(__name__)
logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
logFormat.converter = time.gmtime
logHandler = logging.StreamHandler(sys.stdout)
logHandler.setFormatter(logFormat)
logHandler.setLevel(logging.INFO)
default_log.addHandler(logHandler)
default_log.setLevel(logging.DEBUG)

class Lwa352CorrelatorControl():
    """
    **Description**
    
    A class to encapsulate control of a multi-pipeline LWA correlator system.
    Internally, this instantiates multiple ``Lwa352PipelineControl`` instances.

    **Instantiation**

    :param hosts: A list of hostnames of servers running DSP pipelines.
        Eg. ``['lxdlwagpu0', 'lxdlwagpu1']``
    :type hosts: list of str

    :param npipeline_per_host: The number of pipeline instances running on each server.
    :type npipeline_per_host: int
    
    :param etcdhost: The hostname of the system running the correlator's
        etcd server
    :type etcdhost: string

    :param log: The logger to which this class should emit log messages.
        The default behaviour is to log to stdout
    :type log: logging.Logger

    :ivar hosts: User-passed ``host``
    :ivar npipeline_per_host: User-passed ``npipeline_per_host``
    :ivar npipeline: Total number of pipelines in the system
    :ivar pipelines: List of ``Lwa352PipelineControl`` control instances for each
        server and pipeline.
    :ivar log: User-passed ``log``
    :ivar etcdhost: User-passed ``etcdhost``

    """
    WAIT_DELAY = 5 #: Wait delay, in seconds, after arming blocks
    ARM_DELAY = 5  #: Delay, in seconds, between arm issue and start time
    def __init__(self, hosts, npipeline_per_host=4,
            etcdhost='etcdv3service', log=default_log):

        self.hosts = hosts
        self.npipeline_per_host = npipeline_per_host
        self.log = log
        self.etcdhost = etcdhost
        self.corr_interface = EtcdCorrControl(
                                  etcdhost=etcdhost,
                                  keyroot_cmd='/cmd/corr/x',
                                  keyroot_mon='/mon/corr/x',
                                  keyroot_resp='/resp/corr/x',
                                  log=log,
                              )

        self.pipelines = []
        for host in hosts:
            for pipeline_id in range(npipeline_per_host):
                try:
                    pl = Lwa352PipelineControl(host=host,
                                                pipeline_id=pipeline_id,
                                                etcdhost=self.corr_interface,
                                                log=log)
                except RuntimeError:
                    self.log.error('%s pipeline %d was unresponsive and will be ignored' % (host, pipeline_id))
                    continue
                self.pipelines += [pl]
        self.npipeline = len(self.pipelines)

    def __del__(self):
        for pl in self.pipelines:
            del(pl)
        self.corr_interface.close()
       
    def start_pipelines(self, wait=True, timeout=60*3):
        """
        Start all pipelines, using the default configuration.

        :param wait: If True, wait until the pipelines look like they
            are up and receiving data before returning.
        :type wait: bool

        :param timeout: Timeout, in seconds, to wait for pipelines to
            come up. After this time, issue a warning and return.
        :type timeout: float
        """
        for pl in self.pipelines:
            pl.start_pipeline()

        t0 = time.time()
        if wait:
            while(True):
                try:
                    time.sleep(1)
                    ready = self.pipelines_are_up()
                    if ready:
                        time.sleep(2) # paranoia
                        self.log.info("Pipelines all appear to be ready after %.1f seconds" % (time.time() - t0))
                        return
                except:
                    pass
                if time.time() - t0 > timeout:
                    self.log.warning("Timeout waiting for pipelines to come up after %.1f seconds" % timeout)
                    return

    def pipelines_are_up(self, age_threshold=10, verbose=False):
        """
        Returns True if all pipelines look like they have published recent status data.

        :param age_threshold: The age threshold, in seconds, above which status data
            are considered stale.
        :type age_threshold: float

        :param verbose: If True, print which pipelines are up
        :type verbose: bool
        """
        up = True
        for pl in self.pipelines:
            this_pl_up = pl.pipeline_is_up(age_threshold=age_threshold)
            if verbose:
                print('%s:%d - up? %s' % (pl.host, pl.pipeline_id, this_pl_up))
            up = up and this_pl_up
        return up

    def stop_pipelines(self):
        """
        Stop all pipelines.
        """
        for pl in self.pipelines:
            pl.stop_pipeline(force=True) #Will kill all pipelines on the remote servers
        time.sleep(10)
        stopped = True
        for pl in self.pipelines:
            stopped = stopped and not (pl.pipeline_is_up(age_threshold=10))
        if not stopped:
            self.log.warning("Pipeline %s:%d still seems to be running" % (pl.host, pl.pipeline_id))
    
    def _arm_and_wait(self, blocks, delay):
        """
        Arm blocks in a pipeline.

        :param blocks: blocks to be armed
        :type blocks: list of BlockControl

        :param delay: Duration until arm time, in second. Must
            be at least 5 seconds.
        :type delay: int
        """
        assert delay >= 5, "I won't arm <5 seconds in the future."
        corr_arm_time = blocks[0].get_next_allowed_start(delay)
    
        for b in blocks:
            b.set_start_time(corr_arm_time)
        time.sleep(1)
        
        ok = True
        for b in blocks:
            if b.get_bifrost_status(user_only=True)['state'] != 'waiting':
                ok = False
                self.log.warning("Pipeline %s:%d not in expected waiting state after arm" %
                        (b.host, b.pipeline_id))
    
        if ok:
            self.log.info("All pipelines in 'waiting' state as expected")
    
        wait_time = delay + self.WAIT_DELAY
        self.log.info("Waiting %d seconds for trigger" % wait_time)
        time.sleep(wait_time) # Trigger should have happened by now
                          
        ok = True
        for b in blocks:
            if b.get_bifrost_status(user_only=True)['state'] != 'running':
                ok = False
                self.log.warning("Pipeline %s:%d not in expected running state" %
                        (b.host, b.pipeline_id))
    
        if ok:
            self.log.info("All pipelines in 'running' state as expected")

        return ok
        
    def configure_corr(self, dest_ip='10.41.0.19', dest_port=10001, max_mbps=20000):
        """
        Configure correlator "slow" output, and arm upstream correlation / accumulation
        logic to being outputting data.

        :param dest_ip: Hostname or IP address to which slow correlator packets
            should be sent.
            If list of ips is provided, pipeline ``n`` will use destination ip
            ``dest_ip[n % len(dest_ip)]``
        :type dest_ip: str

        :param dest_port: UDP port to which slow correlator packets should be sent.
            If list of ports is provided, pipeline ``n`` will use destination port
            ``dest_port[n % len(dest_port)]``
        :type dest_port: int or list of int

        :param max_mbps: Total correlator output data rate in Mbits/s. This rate is
            shared over multiple pipelines in the system.
        :type max_mbps: int
        """
        if not isinstance(dest_port, list):
            dest_port = [dest_port]
        if not isinstance(dest_ip, list):
            dest_ip = [dest_ip]

        dest_ip_res = []
        for ip in dest_ip:
            try:
                dest_ip_res += [socket.gethostbyname(ip)]
            except:
                self.log.exception("Couldn't convert hostname %s to IP" % dest_ip)
                raise

        max_mbps_per_pl = max_mbps // self.npipeline
        self.log.info("Setting max data output rate per pipeline to %.1f Mbit/s" % max_mbps_per_pl)

        for pn, pl in enumerate(self.pipelines):
            pl_dest_ip = dest_ip[pn % len(dest_ip)]
            pl_dest_port = dest_port[pn % len(dest_port)]
            self.log.info("Setting pipeline %s:%d data destination to %s:%d" %
                    (pl.host, pl.pipeline_id, pl_dest_ip, pl_dest_port))
            pl.corr_output_full.set_max_mbps(max_mbps_per_pl)
            pl.corr_output_full.set_destination(dest_ip=pl_dest_ip, dest_port=pl_dest_port)

        self.log.info("Arming correlator core")
        self._arm_and_wait([pl.corr for pl in self.pipelines], self.ARM_DELAY)
        self.log.info("Arming correlator accumulator core")
        self._arm_and_wait([pl.corr_acc for pl in self.pipelines], self.ARM_DELAY)

    def plot_autocorrs(self):
        from matplotlib import pyplot as plt
        for p in self.pipelines:
            p.corr_output_full.enable_autos()
        t0 = time.time()
        ready = False
        TIMEOUT = 10
        while(not ready):
            ready = True
            for p in self.pipelines:
                d = p.corr_output_full.get_status()
                if 'autocorr' not in d:
                    ready = False
            if time.time() > t0 + TIMEOUT:
                self.log.error('Timed out after waiting 60s for autocorrs')
                break

        autocorrs = None
        for p in self.pipelines:
            d = p.corr_output_full.get_status()
            if 'autocorr' not in d:
                continue
            ac_dict = json.dumps(d['autocorr'])
            data = ac_dict['data']
            nstand, npol, nchan = data.shape
            chan0 = ac['chan0']
            t = ac['time']
            self.log.info('Got autocorr from pipeline %s:%d for integration %d' % (p.host, p.pipeline_id, t))
            if autocorrs is None:
                autocorrs = np.zeros([nstand, npol, NCHAN])
            autocorrs[:,:,chan0:chan0+nchan] = data

        for i in range(11):
            for j in range(32):
                plt.subplot(11,32,32*i+j+1)
                plt.plot(autocorrs[32*i + j, 0])
                plt.plot(autocorrs[32*i + j, 1])
        plt.show()

class Lwa352PipelineControl():
    """
    **Description**
    
    A class to encapsulate control of a complete LWA correlator/beamformer
    pipeline running on a server.

    **Instantiation**

    :param host: The hostname of the server running the DSP pipeline
        to be commanded
    :type host: string
    :param pipeline_id: The index of the pipeline on this server to be
        commanded
    :type pipeline_id: int
    
    :param etcdhost: The hostname of the system running the correlator's
        etcd server, or, a pre-instantiated EtcdCorrControl instance
    :type etcdhost: string or EtcdCorrControl

    :param log: The logger to which this class should emit log messages.
        The default behaviour is to log to stdout
    :type log: logging.Logger

    :ivar host: User-passed ``host``
    :ivar pipeline_id: User-passed ``pipeline_id``
    :ivar log: User-passed ``log``
    :ivar corr_interface: ``EtcdCorrControl`` instance connected to ``etcdhost``

    :ivar corr: Interface to correlation block, used to control start time
    :ivar corr_acc: Interface to correlation integration block, used to
        control integration length and start time
    :ivar corr_subsel: Interface to baseline subselection block
    :ivar corr_output_full: Interface to full baseline output packetizer
    :ivar corr_output_part: Interface to baseline subselection output packetizer

    :ivar beamform: Interface to beamforming processor
    :ivar beamform_output: Interface to power-beam packetizater
    :ivar beamform_vlbi_output: Interface to voltage-beam packetizer

    :ivar triggered_dump: Interface to triggered buffer dump
    """
    def __init__(self, host='lxdlwagpu02', pipeline_id=0, 
                 etcdhost='etcdv3service', log=default_log):
        self.host = host
        self.pipeline_id = pipeline_id
        self.log = log
        self._corr_interface_from_parent = False
        if isinstance(etcdhost, EtcdCorrControl):
            self.corr_interface = etcdhost
            self._corr_interface_from_parent = True
        else:
            self.corr_interface = EtcdCorrControl(
                                      etcdhost=etcdhost,
                                      keyroot_cmd='/cmd/corr/x',
                                      keyroot_mon='/mon/corr/x',
                                      keyroot_resp='/resp/corr/x',
                                      log=log,
                                  )

        args = [self.log, self.corr_interface, self.host, self.pipeline_id]

        self.capture = BlockControl(*args, name='udp_verbs_capture')
        self.corr_output_full = CorrOutputFullControl(*args)
        self.corr_output_part = CorrOutputPartControl(*args)
        self.corr = CorrControl(*args)
        self.corr_acc = CorrAccControl(*args)
        self.corr_subsel = CorrSubselControl(*args)
        self.triggered_dump = TriggeredDumpControl(*args)
        self.beamform = BeamformControl(*args)
        self.beamform_output = BeamformOutputControl(*args)
        self.beamform_vlbi_output = BeamformVlbiOutputControl(*args)
        if not self.check_connection():
            if not self._corr_interface_from_parent:
                self.corr_interface.close()
            raise RuntimeError("Connection failed. Consider restarting lwa-xeng-etcd.service daemon on host %s" % host)

    def __del__(self):
        if not self._corr_interface_from_parent:
            self.corr_interface.close()

    def start_pipeline(self):
        """
        Start the pipeline, using the default configuration.
        """
        self.corr_interface.send_command(self.host, cmd='start_pipeline', block='xctrl', xid=self.pipeline_id)

    def stop_pipeline(self, force=False):
        """
        Stop the pipeline.

        :param force: If True, brutally "killall" pipelines on the server, rather than trying to kill by PID.
            This can have the adverse effect of killing multiple pipelines unintentionally.
        :type force: bool
        """
        self.corr_interface.send_command(self.host, cmd='stop_pipeline', block='xctrl', xid=self.pipeline_id, force=force)

    def check_connection(self, timeout=1):
        """
        Send a "get pipeline PID" command to the pipeline control interface to see if it is alive.

        :param timeout: The length of time, in seconds, to wait for a response from the pipeline.
        :type timeout: float

        :return: True, if connected. Else, False
        :rtype: bool
        """
        try:
            self.corr_interface.send_command(self.host, cmd='get_pid', block='xctrl', xid=self.pipeline_id, timeout=timeout)
            return True
        except RuntimeError:
            self.log.error("Control interface on host %s failed to respond to ping!" % self.host)
            return False

    def pipeline_is_up(self, age_threshold=10):
        """
        Returns True if the pipeline looks like it has published recent status data.

        :param age_threshold: The age threshold, in seconds, above which status data
            are considered stale.
        :type age_threshold: float
        """
        try:
            if time.time() - self.corr.get_bifrost_status()['time'] < age_threshold:
                return True
        except:
            pass
        return False
