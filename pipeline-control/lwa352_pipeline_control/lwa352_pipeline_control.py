import argparse
import time
import sys
import logging

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
logHandler.setLevel(logging.DEBUG)
default_log.addHandler(logHandler)
default_log.setLevel(logging.DEBUG)

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
    def __init__(self, host='rtr-dev2', pipeline_id=0, 
                 etcdhost='etcdhost', log=default_log):
        self.host = host
        self.pipeline_id = pipeline_id
        self.log = log
        self.corr_interface = EtcdCorrControl(
                                  etcdhost=etcdhost,
                                  keyroot_cmd='/cmd/corr/x',
                                  keyroot_mon='/mon/corr/x',
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
        


