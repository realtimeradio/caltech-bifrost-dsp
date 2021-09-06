#! /usr/bin/env python

import time
import argparse
import logging
import logging.handlers
from lwa352_pipeline_control import lwa352_xeng_etcd_client

def main():
    parser = argparse.ArgumentParser(description='Start an ETCD X-Engine control service',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', dest='etcdhost', type=str, default='etcdv3service.sas.pvt',
                        help ='Host serving etcd')
    parser.add_argument('-p', dest='port', type=int, default=2379,
                        help ='Port on which etcd is served')
    args = parser.parse_args()

    logger = logging.getLogger("%s" % (__file__))
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)20s - %(levelname)s - %(message)s')
    handler = logging.handlers.SysLogHandler(address='/dev/log')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    logger.info("Starting ETCD client service")
    try:
        ec = lwa352_xeng_etcd_client.LwaXengineEtcdClient(
                etcdhost=args.etcdhost,
                etcdport=args.port,
                logger=logger,
            )
    except RuntimeError:
        logger.exception("Failed to instantiate etcd control service")
        raise

    ec.start_command_watch()

    while(True):
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
