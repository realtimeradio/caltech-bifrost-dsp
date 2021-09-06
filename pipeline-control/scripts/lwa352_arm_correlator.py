import argparse
import time
from lwa352_pipeline_control import Lwa352CorrelatorControl

def main(args):
    corr = Lwa352CorrelatorControl(args.hosts, args.pipelines, args.etcdhost)
    corr.ARM_DELAY = args.delay
    corr.configure_corr(args.destip, args.destport, args.maxmbps*corr.npipeline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Display perfomance of blocks in Bifrost pipelines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-p', '--pipelines', type=int, default=4,
                        help='Number of pipelines per host')
    parser.add_argument('hosts', nargs='+',
                        help='Hostnames of servers running pipelines')
    parser.add_argument('-d', '--delay', type=int, default=5,
                        help='How long in future to arm correlator')
    parser.add_argument('--destip', type=str, default='10.41.0.19',
                        help='Destination IP for slow correlation data')
    parser.add_argument('--destport', type=int, default=10001,
                        help='Destination UDP port for slow correlation data')
    parser.add_argument('--maxmbps', type=int, default=1500,
                        help='Maximum Mbits/s output for each pipeline')
    parser.add_argument('--etcdhost', default='etcdv3service.sas.pvt',
                        help='etcd host to which stats should be published')
    args = parser.parse_args()
    main(args)
