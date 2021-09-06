import argparse
import time
import sys
import logging
import socket

from .etcd_control import EtcdCorrControl

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

        self.pipelines = []
        for host in hosts:
            for pipeline_id in range(npipeline_per_host):
                self.pipelines += [Lwa352PipelineControl(host=host,
                                                            pipeline_id=pipeline_id,
                                                            etcdhost=etcdhost,
                                                            log=log,
                                                        )]
        self.npipeline = len(self.pipelines)
       
    def start_pipelines(self):
        """
        Start all pipelines, using the default configuration.
        """
        for pl in self.pipelines:
            pl.start_pipeline()
        time.sleep(10)

    def stop_pipelines(self):
        """
        Stop all pipelines.
        """
        for pl in self.pipelines:
            pl.stop_pipeline()
        time.sleep(5)
    
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
        etcd server
    :type etcdhost: string

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
        self.corr_interface = EtcdCorrControl(
                                  etcdhost=etcdhost,
                                  keyroot_cmd='/cmd/corr/x',
                                  keyroot_mon='/mon/corr/x',
                                  keyroot_resp='/resp/corr/x',
                                  log=log,
                              )

        args = [self.log, self.corr_interface, self.host, self.pipeline_id]

        self.corr_output_full = CorrOutputFullControl(*args)
        self.corr_output_part = CorrOutputPartControl(*args)
        self.corr = CorrControl(*args)
        self.corr_acc = CorrAccControl(*args)
        self.corr_subsel = CorrSubselControl(*args)
        self.triggered_dump = TriggeredDumpControl(*args)
        self.beamform = BeamformControl(*args)
        self.beamform_output = BeamformOutputControl(*args)
        self.beamform_vlbi_output = BeamformVlbiOutputControl(*args)

    def start_pipeline(self):
        """
        Start the pipeline, using the default configuration.
        """
        self.corr_interface.send_command(self.host, cmd='start_pipeline', block='xctrl', xid=self.pipeline_id)

    def stop_pipeline(self):
        """
        Start the pipeline.
        """
        self.corr_interface.send_command(self.host, cmd='stop_pipeline', block='xctrl', xid=self.pipeline_id)

