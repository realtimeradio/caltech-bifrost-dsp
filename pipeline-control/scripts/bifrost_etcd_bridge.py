#!/usr/bin/env python

import argparse
import socket
import glob
import os
import time
import re

import simplejson as json
import etcd3 as etcd
from bifrost.proclog import load_by_pid

BIFROST_STATS_BASE_DIR = '/dev/shm/bifrost/'

def get_command_line(pid):
    """
    Given a PID, use the /proc interface to get the full command line for 
    the process.  Return an empty string if the PID doesn't have an entry in
    /proc.
    """

    cmd = ''
    try:
        with open('/proc/%i/cmdline' % pid, 'r') as fh:
            cmd = fh.read()
            cmd = cmd.replace('\0', ' ')
            fh.close()
    except IOError:
        pass
    return cmd

def poll(base_dir):
    ## Find all running processes
    pidDirs = glob.glob(os.path.join(base_dir, '*'))
    pidDirs.sort()

    ## Load the data
    blockList = {}
    for pn, pidDir in enumerate(pidDirs):
        pid = int(os.path.basename(pidDir), 10)
        contents = load_by_pid(pid)

        cmd = get_command_line(pid)
        if cmd == '':
            continue

        # Get a pipeline ID from the first block willing to provide one
        pipeline_id = None
        for block in contents.keys():
            try:
                pipeline_id = contents[block]['sequence0']['pipeline_id']
            except KeyError:
                continue
        if pipeline_id is None:
            pipeline_id = pn

        for block in contents.keys():
            try:
                log = contents[block]['bind']
                cr = log['core0']
            except KeyError:
                continue

            try:
                log = contents[block]['perf']
                ac = max([0.0, log['acquire_time']])
                pr = max([0.0, log['process_time']])
                re = max([0.0, log['reserve_time']])
                gb = max([0.0, log.get('gbps', 0.0)])
            except KeyError:
                ac, pr, re, gb = 0.0, 0.0, 0.0, 0.0

            blockList['%i-%s' % (pipeline_id, block)] = {
                'pid': pid, 'name':block, 'cmd': cmd, 'core': cr,
                'acquire': ac, 'process': pr, 'reserve': re, 'total':ac+pr+re,
                'gbps':gb, 'time':time.time()}

            try:
                log = contents[block]['sequence0']
                blockList['%i-%s' % (pipeline_id, block)].update(log)
            except:
                pass


            # Get User stats
            try:
                if 'stats' in contents[block]:
                    log = contents[block]['stats']
                    for k, v in log.items():
                        if v == 'True':
                            log[k] = True
                        elif v == 'False':
                            log[k] = False
                    blockList['%i-%s' % (pipeline_id, block)]['stats'] = log
            except:
                print("Error parsing stats")

    return time.time(), blockList

def main(args):
    ec = etcd.client(args.etcdhost)
    last_poll = 0
    # historical bytes captured, keyed by PID
    capture_times = {} # store the last time we captured bytes received
    capture_bytes = {} # store the last number of bytes received captured
    while True:
        try:
            wait_time = max(0, last_poll + args.polltime - time.time())
            time.sleep(wait_time)
            last_poll, d = poll(BIFROST_STATS_BASE_DIR)
            for k, v in d.items():
                pipeline_id, block = k.split('-')
                # If the block name ends in _<number> (which seem to be how
                # bifrost handles multiple blocks of the same type, figure
                # out the block id.
                # Bifrost labels block 1: "Block", block 2: "Block_2" etc
                # Seach for _<number> and if it exists, strip it off and
                # use the value to calculate a block id.
                x = re.search(r'_\d+$', block)
                if x is not None:
                    block_id = int(x.group()[1:]) - 1 # convert to 0-indexing
                    block = block.rstrip(x.group())
                else:
                    block_id = 0
                # Special case to handle capture capture gbps
                if block == "udp_verbs_capture":
                    try:
                        last_capture_bytes = capture_bytes.get(pipeline_id, 0)
                        last_capture_time  = capture_times.get(pipeline_id, 0)
                        this_capture_bytes = v['stats'].get('ngood_bytes', 0)
                        this_capture_time  = last_poll
                        gbps = (this_capture_bytes - last_capture_bytes) / (this_capture_time - last_capture_time) * 8 / 1e9
                        capture_times[pipeline_id] = this_capture_time
                        capture_bytes[pipeline_id] = this_capture_bytes
                        v['gbps'] = gbps
                    except:
                        pass
                # Special case to remove lots of baseline selection status traffic. TODO: how should we read this?
                if block == "CorrSubsel":
                    try:
                        v['stats'].pop("baselines")
                    except:
                        pass
                
                ekey = '{keybase}/x/{hostbase}/pipeline/{pipeline_id}/{block}/{block_id}'.format(
                           keybase=args.keybase,
                           hostbase=args.hostbase,
                           pipeline_id=pipeline_id,
                           block=block,
                           block_id=block_id,
                       )
                print(ekey)
                ec.put(ekey, json.dumps(v))
            
        except KeyboardInterrupt:
           break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Display perfomance of blocks in Bifrost pipelines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--etcdhost', default='etcdhost',
                        help='etcd host to which stats should be published')
    parser.add_argument('--keybase', default='/mon/corr',
                        help='Key to which stats should be published: '
                             '<keybase>/x/<hostbase>/pipeline/<pipeline-id>/blockname/...')
    parser.add_argument('--hostbase', default=socket.gethostname(),
                        help='Key to which stats should be published: '
                        '<keybase>/x/<hostbase>/pipeline/<pipeline-id>/blockname/...')
    parser.add_argument('-t', '--polltime', type=int, default=10,
                        help='How often to poll stats, in seconds')
    args = parser.parse_args()
    main(args)
