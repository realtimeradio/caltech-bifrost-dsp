#!/usr/bin/env python

import os
import sys
import glob
import time
import curses
import socket
import argparse
import traceback
import subprocess
from io import StringIO
import simplejson as json
import etcd3 as etcd


def _add_line(screen, y, x, string, *args):
    """
    Helper function for curses to add a line, clear the line to the end of 
    the screen, and update the line number counter.
    """

    screen.addstr(y, x, string, *args)
    screen.clrtoeol()
    return y + 1


_REDRAW_INTERVAL_SEC = 0.2

def add_indented_lines(d, indent_level, blacklist):
    """
    given an input dictionary, output
    an hierarchically indented string
    """
    s = ""
    indent = indent_level * "  "
    for k, v in sorted(d.items()):
       if k in blacklist:
           continue
       if isinstance(v, dict):
           s += "%s%s:\n" % (indent, k)
           s += add_indented_lines(v, indent_level + 1, blacklist)
       else:
           s += "%s%s: %s\n" % (indent, k, v) 
    return s

def gen_indented_list(d, indent_level, blacklist):
    """
    given an input dictionary, output
    an hierarchically indented string
    """
    out = []
    for k, v in sorted(d.items()):
       if k in blacklist:
           continue
       if isinstance(v, dict):
           out += [{'indent':indent_level, 'key':k}]
           out += gen_indented_list(v, indent_level + 1, blacklist)
       else:
           out += [{'indent':indent_level, 'key':k, 'val':v}]
    return out

def make_hier_dict(din):
    """
    Given a dictionary, `din` with keys like
    a/b/c:x, a/b/d:y, ...
    return a hierarchical dictionary like:
    {a: b: {c:x, d:y}}
    """
    def add_to_dict(d, x):
        for k,v in x.items():
            levels = k.split('/')
            if len(levels) == 1:
                d[levels[0]] = v
            else:
                upper = levels[0]
                lower = levels[1:]
                if upper not in d:
                    d[upper] = {}
                add_to_dict(d[upper], {'/'.join(lower):v})
    out = {}
    add_to_dict(out, din)
    return out


def main(args):

    try:
        ec = etcd.client(args.etcdhost)
    except:
        print('Failed to connect to ETCD host %s' % args.etcdhost)
        exit()

    scr = curses.initscr()
    curses.start_color()

    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_RED)
    RED = curses.color_pair(1)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_YELLOW)
    ORANGE = curses.color_pair(2)

    curses.noecho()
    curses.cbreak()
    scr.keypad(1)
    scr.nodelay(1)
    size = scr.getmaxyx()

    std = curses.A_NORMAL
    rev = curses.A_REVERSE


    poll_interval = 1.0
    tLastPoll = 0.0

    def highlight_warnings(x):
        if x['key'] == 'timestamp':
            if x['val'] < time.time() - 3: return RED
        elif x['key'] == 'gbps':
            if x['val'] < 17.0: return RED
            elif x['val'] < 25.0: return ORANGE
        return std
    
    hn = 0
    try:
        while True:
            update = (time.time() - tLastPoll > poll_interval)

            ## Interact with the user
            c = scr.getch()
            curses.flushinp()
            if c == ord('q'):
                break
            # If the user wants stats from a different pipeline then
            # force a data update
            elif c == ord('n'):
                hn += 1
                update = True
            elif c == ord('p'):
                hn -= 1
                update = True

            ## Do we need to poll the system again?
            # Get all the valid keys
            # Decode JSON and recast list of keys into a hierarchical dictionary
            if update:
                tLastPoll = time.time()
                # Generate a list of hosts (assuming keys have format keyroot/<host>/...)
                hosts = [_[1].key.decode()[len(args.keyroot):].lstrip('/').split('/')[0] for _ in ec.get_prefix(args.keyroot, keys_only=True)]
                n_hosts = len(hosts)
                # Use the updateable host-number value to decide which stats to show
                host = hosts[hn % n_hosts]
                d = {}
                for (s, meta) in ec.get_prefix(args.keyroot + '/' + host + '/'):
                    k = meta.key.decode()[len(args.keyroot):]
                    try:
                        d[k] = json.loads(s)
                    except:
                        d[k] = 'JSON_DECODE_FAIL'
                data = make_hier_dict(d)

            ## Display
            k = 0
            ### General - load average
            for key, v in sorted(data.items()):
                #output = add_indented_lines(v, 0, args.keyblacklist.split(','))
                #_add_line(scr, k, 0, output, std)
                #k += len(output.split('\n'))
                output = gen_indented_list(v, 0, args.keyblacklist.split(','))
                for line in output:
                    s = line['key']+':'
                    if 'val' in line:
                        s += ' %s' % line['val']
                        mode = highlight_warnings(line)
                    else:
                        mode = std
                    k = _add_line(scr, k, 2*line['indent'], s, mode)

            k += 4
            k = _add_line(scr, k, 0, time.ctime(), std)
            _add_line(scr, k, 0, '%d'%hn, std)
            ### Clear to the bottom
            scr.clrtobot()
            ### Refresh
            scr.refresh()

            ## Sleep
            time.sleep(_REDRAW_INTERVAL_SEC)

    except KeyboardInterrupt:
        pass

    except Exception as err:
        error = err
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fileObject = StringIO()
        traceback.print_tb(exc_traceback, file=fileObject)
        tbString = fileObject.getvalue()
        fileObject.close()

    # Save the window contents
    contents = ''
    y,x = scr.getmaxyx()
    for i in range(y-1):
        for j in range(x):
            d = scr.inch(i,j)
            c = d&0xFF
            a = (d>>8)&0xFF
            contents += chr(c)

    # Tear down curses
    scr.keypad(0)
    curses.echo()
    curses.nocbreak()
    curses.endwin()
    
    # Final reporting
    try:
        ## Error
        print("%s: failed with %s" % (os.path.basename(__file__), str(error)))
        for line in tbString.split('\n'):
            print(line)
    except NameError:
        ## Last window contents sans attributes
        print(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Display perfomance of different blocks of Bifrost pipelines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--etcdhost', default='localhost',
                        help='etcd host to which stats should be published')
    parser.add_argument('--keyroot', default='/mon/corr',
                        help='Base key directory to watch.')
    parser.add_argument('--keyblacklist', default='cmd,name,pid',
                        help='Comma separated list of keys to ignore')
    args = parser.parse_args()
    main(args)
    
