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
    def __init__(self, etcdhost='etcdhost', keyroot_cmd='/cmd/corr/x', keyroot_mon='/mon/corr/x', log=default_log):
        self.keyroot_cmd = keyroot_cmd
        self.keyroot_mon = keyroot_mon
        self.etcdhost = etcdhost
        self.log = log
        try:
            self.ec = etcd.client(self.etcdhost)
        except:
            log.error('Failed to connect to ETCD host %s' % self.etcdhost)
            raise

    def _get_cmd_key(self, host, pipeline, block):
        return self.keyroot_cmd + '/%s/pipeline/%d/%s/ctrl' % (host, pipeline, block)

    def _get_mon_key(self, host, pipeline, block):
        return self.keyroot_mon + '/%s/pipeline/%d/%s' % (host, pipeline, block)

    def send_command(self, host, pipeline, block, **kwargs):
        key = self._get_cmd_key(host, pipeline, block)
        val = json.dumps(kwargs)
        self.ec.put(key, val)

    def get_status(self, host, pipeline, block, user_only=True):
        key = self._get_mon_key(host, pipeline, block)
        val, meta = self.ec.get(key)
        val = json.loads(val)
        if user_only:
            return val.get("stats", {})
        else:
            return val


