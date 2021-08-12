import argparse
import time
from lwa352_pipeline_control import Lwa352PipelineControl

WAIT_DELAY = 5

def arm_and_wait(blocks, delay):
    assert delay >= 5, "I won't arm <5 seconds in the future."
    corr_arm_time = blocks[0].get_next_allowed_start(delay)

    for b in blocks:
        b.set_start_time(corr_arm_time)
    time.sleep(1)
    
    ok = True
    for b in blocks:
        if b.get_bifrost_status(user_only=True)['state'] != 'waiting':
            ok = False
            print("Pipeline %s:%d not in expected waiting state after arm" %
                    (b._host, b._pipeline_id))

    if ok:
        print("All pipelines in 'waiting' state as expected")

    wait_time = delay + WAIT_DELAY
    print("Waiting %d seconds for trigger" % wait_time)
    time.sleep(wait_time) # Trigger should have happened by now
                      
    ok = True
    for b in blocks:
        if b.get_bifrost_status(user_only=True)['state'] != 'running':
            ok = False
            print("Pipeline %s:%d not in expected running state" %
                    (b._host, b._pipeline_id))

    if ok:
        print("All pipelines in 'running' state as expected")

def main(args):
    controllers = []
    for host in args.hosts:
        for p in range(args.pipelines):
            controllers += [Lwa352PipelineControl(
                               host=host,
                               pipeline_id=p,
                               etcdhost=args.etcdhost)
                           ]

    # First set up destination
    for c in controllers:
        c.corr_output_full.set_max_mbps(args.maxmbps)
        c.corr_output_full.set_destination(dest_ip=args.destip, dest_port=args.destport)

    # First arm fast correlator
    arm_and_wait([c.corr for c in controllers], args.delay)
    # Next arm accumulator
    arm_and_wait([c.corr_acc for c in controllers], args.delay)


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
