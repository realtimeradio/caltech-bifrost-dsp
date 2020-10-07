import argparse
import time
import sys
import logging

from etcd_control import EtcdCorrControl

from blocks.corr_output_full_control import CorrOutputFull
from blocks.corr_output_part_control import CorrOutputPart
from blocks.corr_acc_control import CorrAcc
from blocks.corr_control import Corr
from blocks.corr_subsel_control import CorrSubsel

default_log = logging.getLogger(__name__)
logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
logFormat.converter = time.gmtime
logHandler = logging.StreamHandler(sys.stdout)
logHandler.setFormatter(logFormat)
logHandler.setLevel(logging.DEBUG)
default_log.addHandler(logHandler)
default_log.setLevel(logging.DEBUG)

class Lwa352XControl():
    def __init__(self, host='rtr-dev2', pipeline_id=0, etcdhost='etcdhost', log=default_log):
        self.host = host
        self.pipeline_id = pipeline_id
        self.log = log
        self.corr_interface = EtcdCorrControl(etcdhost=etcdhost, keyroot_cmd='/cmd/corr/x', keyroot_mon='/mon/corr/x', log=log)

        self.corr_output_full = CorrOutputFull(self.log, self.corr_interface, self.host, self.pipeline_id)
        self.corr_output_part = CorrOutputPart(self.log, self.corr_interface, self.host, self.pipeline_id)
        self.corr = Corr(self.log, self.corr_interface, self.host, self.pipeline_id)
        self.corr_acc = CorrAcc(self.log, self.corr_interface, self.host, self.pipeline_id)
        self.corr_subsel = CorrSubsel(self.log, self.corr_interface, self.host, self.pipeline_id)
        


