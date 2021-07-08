import argparse
from lwa352_pipeline_control import Lwa352PipelineControl

def arm_and_wait(blocks, delay):
    for b in blocks:
        b.set_start_time(corr_arm_time)
    
    for b in blocks:
        if b.get_bifrost_status(user_only=True)['state'] != 'waiting':
            print("Pipeline %s:%d not in expected waiting state after arm" %
                    (b._host, b._pipeline_id))

    wait_time = args.delay + 3
    print("Waiting %d seconds for trigger" % wait_time)
    time.sleep(wait_time) # Trigger should have happened by now
                      
    for b in blocks:
        if b.get_bifrost_status(user_only=True)['state'] != 'running':
            print("Pipeline %s:%d not in expected running state" %
                    (b._host, b._pipeline_id))

def main(args):
    controllers = []
    for host in args.hosts:
        for p in args.pipelines:
            controllers += [Lwa352PipelineControl(
                               host=host,
                               pipeline_id=p,
                               etcdhost=args.etcdhost)
                           ]
    # First arm fast correlator
    corr_arm_time = controllers[0].corr.get_next_allowed_start(args.delay)
    arm_and_wait([c.corr for c in controllers], args.delay)
    # Next arm accumulator
    corr_arm_time = controllers[0].corr_acc.get_next_allowed_start(args.delay)
    arm_and_wait([c.corracc for c in controllers], args.delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Display perfomance of blocks in Bifrost pipelines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-p', '--pipelines', type=int, default='2',
                        help='Number of pipelines per host')
    parser.add_argument('hosts',
                        help='Hostnames of servers running pipelines')
    parser.add_argument('-d', '--delay', type=int, default=10,
                        help='How long in future to arm correlator')
    parser.add_argument('--etcdhost', default='etcdv3service.sas.pvt',
                        help='etcd host to which stats should be published')
    args = parser.parse_args()
    main(args)
