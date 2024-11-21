import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.packet_writer import DiskWriter, HeaderInfo

import time
import ujson as json
import numpy as np
import os
import mmap
import struct
import datetime

from .block_base import Block

HEADER_SIZE = 1024*1024

LWA352_EPOCH     = datetime.datetime(1970, 1, 1)
LWA352_FS        = 196.0e6
LWA352_CLOCK     = 196.0e6
LWA352_NCHAN     = 4096
LWA352_CHAN_BW   = LWA352_CLOCK / (2*LWA352_NCHAN)

def get_time_tag(dt=datetime.datetime.utcnow(), seq_offset=0):
    timestamp = int((dt - LWA352_EPOCH).total_seconds())
    time_tag  = timestamp*int(LWA352_FS) + seq_offset*(int(LWA352_FS)//int(LWA352_CHAN_BW))
    return time_tag
def seq_to_time_tag(seq):
    return seq*(int(LWA352_FS)//int(LWA352_CHAN_BW))
def time_tag_to_seq_float(time_tag):
    return time_tag*LWA352_CHAN_BW/LWA352_FS

LWA352_DISK_NOOP = False

class DummyFileHandle:
    def __init__(self, fileno):
        self._fileno = fileno
        
    def fileno(self):
        return self._fileno


class TriggeredDump(Block):
    """
    **Functionality**

    This block writes data from an input bifrost ring buffer to disk, when triggered.

    **New Sequence Condition**

    This block is a bifrost sink, and generates no output sequences.

    **Input Header Requirements**

    This block requires that the following header fields be provided by the upstream data
    source:

    .. table::
        :widths: 25 10 10 55

        +-----------+--------+-------+---------------------------------------------------+
        | Field     | Format | Units | Description                                       |
        +===========+========+=======+===================================================+
        | ``seq 0`` | int    |       | Spectra number for the first sample in the input  |
        |           |        |       | sequence                                          |
        +-----------+--------+-------+---------------------------------------------------+

    The field ``seq`` if provided by the upstream block will be overwritten.

    **Output Headers**

    This block is a bifrost sink, and generates no output headers. Headers provided
    by the upstream block are written to this block's data files, with the exception of the
    ``seq`` field, which is overwritten.

    **Data Buffers**

    *Input Data Buffer*: A bifrost ring buffer of at least
    ``nbyte_per_time x ntime_gulp`` bytes size.

    *Output Data Buffer*: None

    **Instantiation**

    :param log: Logging object to which runtime messages should be
        emitted.
    :type log: logging.Logger

    :param iring: bifrost input data ring
    :type iring: bifrost.ring.Ring

    :param guarantee: If True, read data from the input ring in blocking "guaranteed"
        mode, applying backpressure if necessary to ensure no data are missed by this block.
    :type guarantee: Bool

    :param core: CPU core to which this block should be bound. A value of -1 indicates no binding.
    :type core: int

    :param ntime_gulp: Number of time samples to copy with each gulp.
    :type ntime_gulp: int

    :param nbyte_per_time: Number of bytes per time sample. The total number of bytes read
        with each gulp is ``nbyte_per_time x ntime_gulp``.
    :type nbyte_per_time: int

    :parameter etcd_client: Etcd client object used to facilitate control of this block.
        If ``None``, do not use runtime control.
    :type etcd_client: etcd3.client.Etcd3Client

    :parameter dump_path: Root path to directory where dumped data should be stored.
        This parameter can be overridden by runtime control commands.
    :type data_path: string

    :parameter ntime_per_file: Number of time samples of data to write to each output file.
        This parameter can be overridden by runtime control commands.
    :type ntime_per_file: int


    **Runtime Control and Monitoring**

    This block accepts the following command fields:

    .. table::
        :widths: 25 10 10 55

        +------------------+--------+---------+----------------------------------------------------------------------+
        | Field            | Format | Units   | Description                                                          |
        +==================+========+=========+======================================================================+
        | ``command``      | string |         | Commands:                                                            |
        |                  |        |         |                                                                      |
        |                  |        |         | ``Trigger``: Begin capturing data ASAP                               |
        |                  |        |         |                                                                      |
        |                  |        |         | ``Abort``: Abort a capture currently in progress and delete its data |
        |                  |        |         |                                                                      |
        |                  |        |         | ``Stop``: Stop a capture currently in progress                       |
        +------------------+--------+---------+----------------------------------------------------------------------+
        | ``ntime_per      | int    | samples | Number of time samples to capture in each file.                      |
        | _sample``        |        |         |                                                                      |
        +------------------+--------+---------+----------------------------------------------------------------------+
        | ``nfile``        | int    |         | Number of files to capture per trigger event.                        |
        +------------------+--------+---------+----------------------------------------------------------------------+
        | ``dump_path``    | str    |         | Root path to directory where data should be stored.                  |
        +------------------+--------+---------+----------------------------------------------------------------------+

    **Output Data Format**

    When triggered, this block will output a series of ``nfile`` data files, each containing ``ntime_per_file``
    time samples of data.

    File names begin ``lwa-dump-`` and are suffixed by the trigger time as a floating-point UNIX timestamp with
    two decimal points of precision, a file extension ".tbf", and a file number ".N" where N runs from 0 to
    ``nfile - 1``. For example: ``lwa-dump-1607434049.77.tbf.0``.

    The files have the following structure:
      - The first 4 bytes contains a little-endian 32-bit integer, ``hsize``,  describing the number of bytes of
        subsequent header data.
      - The following 4 bytes contains a little-endian 32-bit integer, ``hblock_size`` describing the size of
        header block which precedes the data payload in the file.
      - Bytes ``8`` to ``8 + hsize`` contain the json-encoded header of the bifrost sequence which is contained
        in the payload of this file, with an additional ``seq`` keyword, which indicates the spectra number of the
        first spectra in this data file.
      - Bytes ``hblock_size`` to ``EOF`` contain data in the shape and format of the input bifrost data buffer.

    Output data files can thus be read with:
    
    .. code-block:: python

        import struct
        with open(FILENAME, "rb") as fh:
            hsize = struct.unpack('<I', fh.read(4))[0]
            hblock_size = struct.unpack('<I', fh.read(4))[0]
            header = json.loads(fh.read(hsize))
            fh.seek(hblock_size)
            data = fh.read() # read to EOF


    """

    def __init__(self, log, iring, ntime_gulp=2500, ntime_per_file=1000000,
                 guarantee=True, core=-1, nbyte_per_time=192*352*2, etcd_client=None, dump_path='/tmp'):

        super(TriggeredDump, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        self.ntime_gulp = ntime_gulp
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nbyte_per_time
        self.nbyte_per_time = nbyte_per_time

        self.define_command_key('command', type=str,
                                condition=lambda x: x in ['trigger', 'abort', 'stop'])
        self.define_command_key('ntime_per_file', type=int, initial_val=ntime_per_file)
        self.define_command_key('nfile', type=int, initial_val=1)
        self.define_command_key('dump_path', type=str, initial_val=dump_path,
                                condition=lambda x: os.path.isdir(x))

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        start = False # Should we start?
        last_trigger_time = None
        dump_path = self.command_vals['dump_path']
        ntime_per_file = 0
        nfile = 1
        filename = None
        while not self.iring.writing_ended():
            # Check trigger every few ms
            time.sleep(0.05)
            # If no new commands keep waiting
            if not self.update_pending:
                continue
            self.update_command_vals()
            if self.command_vals['command'] == 'trigger':
                # Shortcut variables
                ntime_per_file = self.command_vals['ntime_per_file']
                nfile = self.command_vals['nfile']
                dump_path = self.command_vals['dump_path']
                last_trigger_time = time.time()
                filename = os.path.join(dump_path, "lwa-dump-%.2f.tbf" % last_trigger_time)
                self.update_stats({'last_trigger_time' : last_trigger_time,
                                   'state' : 'triggering'})
                
                try:
                    self.dump(filename, ntime_per_file, nfile)
                except Exception as e:
                    self.log.error("TRIGGERED DUMP EXCEPTION >> %s", str(e))

    def dump(self, filename, ntime_per_file, nfile):
        # Create a 4kB-aligned 1M buffer to store header data.
        # Needs to be 512-byte aligned to work with O_DIRECT writing.
        # Luckily mmap aligns to memory page size
        hinfo = mmap.mmap(-1, HEADER_SIZE)
        this_time = 0
        ofile = None
        started = False
        file_num = 0
        file_ndumped = 0
        total_bytes = 0
        
        desc = HeaderInfo()
        
        # Come up with a time to dump
        time_offset    = -120.0      # I get that 256 GB is about 170 s so -120 s should be ok
        time_offset_s  = int(time_offset)
        time_offset_us = int(round((time_offset-time_offset_s)*1e6))
        time_offset    = datetime.timedelta(seconds=time_offset_s, microseconds=time_offset_us)
        
        utc_now = datetime.datetime.utcnow()
        dump_time_tag = get_time_tag(utc_now + time_offset)
        self.log.warning("TRIGGERED DUMP >> requesting at time tag %i", dump_time_tag)
        if LWA352_DISK_NOOP:
            self.log.warning("TRIGGERED DUMP >> not writing to disk for this event")
            
        # Don't go back to idle as soon as we start. If
        # there is a break in the sequence we should keep writing
        #start = False
        #file_num = 0
        #file_ndumped = 0
        #total_bytes = 0
        start_time = time.time()
        #with self.iring.open_sequence_at(self.igulp_size*16, guarantee=self.guarantee) as iseq:
        with self.iring.open_earliest_sequence(guarantee=True) as iseq:
            ihdr = json.loads(iseq.header.tostring())
            time_tag0 = ihdr['seq0'] * 2*LWA352_NCHAN
            
            self.sequence_proclog.update(ihdr)
            
            frame_nbyte = self.nbyte_per_time
            dump_seq_offset  = int(time_tag_to_seq_float(dump_time_tag - time_tag0))
            dump_byte_offset = dump_seq_offset * frame_nbyte
            self.log.warning("TRIGGERED DUMP >> opened at %d", time_tag0)
            self.log.warning("TRIGGERED_DUMP >> offsetting by %d sequences", dump_seq_offset)
            self.log.warning("TRIGGERED DUMP >> offsetting by %d bytes", dump_byte_offset)
            
            # Clean out some of the ring
            prev_time = time.time()
            bytes_rpted = 0
            acquire_time = 0
            process_time = 0
            for ispan in iseq.read(self.igulp_size, begin=dump_byte_offset):
                if ispan.size < self.igulp_size:
                    self.log.warning("TRIGGERED DUMP >> got small gulp.")
                    if started:
                        self.log.error("TRIGGERED DUMP >> got small gulp after start.")
                        break
                    continue
                curr_time = time.time()
                acquire_time += curr_time - prev_time
                prev_time = curr_time
                
                if not started:
                    self.log.info("TRIGGERED DUMP >> opened at %d", dump_byte_offset)
                    started = True
                this_time = ihdr['seq0'] + ispan.offset // frame_nbyte
                ihdr['seq'] = this_time
                if ofile is None or file_ndumped >= ntime_per_file:
                    if file_ndumped >= ntime_per_file:
                        # Close file and increment file number
                        if not LWA352_DISK_NOOP:
                            del udt
                            os.close(ofile)
                        ofile = None
                        file_num += 1
                    if file_num == nfile:
                        self.log.info("TRIGGERED DUMP >> File writing ended")
                        self.update_stats({'status'       : 'complete'})
                        break
                    # If we're here we need to open a new file
                    # Doing the writing from python with directIO -- yikes!
                    # We can only write to such a file with 512-byte aligned
                    # access. Bifrost should guarantee this if it was compiled with
                    # such an option
                    self.update_stats({'status' : 'writing'})
                    self.log.info("TRIGGERED DUMP >> Opening %s" % (filename + '.%d' % file_num))
                    self.log.info("Start seq is %d" % this_time)
                    file_ndumped = 0
                    if not LWA352_DISK_NOOP:
                        ofile = os.open(
                                    filename + '.%d' % file_num,
                                    os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_DIRECT | os.O_SYNC,
                                )
                    else:
                        ofile = 1
                    header = json.dumps(ihdr).encode()
                    hsize = len(header)
                    hinfo.seek(0)
                    hinfo.write(struct.pack('<2I', hsize, HEADER_SIZE) + json.dumps(ihdr).encode())
                    if not LWA352_DISK_NOOP:
                        os.write(ofile, hinfo)
                        dfh = DummyFileHandle(ofile)
                        udt = DiskWriter('generic_%d' % (self.ntime_gulp//4*frame_nbyte), dfh, core=self.core)
                        
                # Write the data
                if not LWA352_DISK_NOOP:
                    idata = ispan.data.reshape(4, 1, self.ntime_gulp//4*frame_nbyte)
                    #os.write(ofile, ispan.data)
                    udt.send(desc, 1, 1, 1, 1, idata)
                file_ndumped += self.ntime_gulp
                total_bytes += self.igulp_size
                bytes_rpted += self.igulp_size
                self.update_stats({'bytes_dumped'  : total_bytes,
                                   'files_created' : file_num+1})
                
                curr_time = time.time()
                process_time += curr_time - prev_time
                prev_time = curr_time
                if bytes_rpted > 1e9:
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': -1, 
                                              'process_time': process_time,
                                              'gbps': 8*bytes_rpted / process_time / 1e9})
                                              
                    bytes_rpted = 0
                    acquire_time = 0
                    process_time = 0
                    
                # If no new commands, loop again
                if not self.update_pending:
                    continue
                self.update_command_vals()
                if self.command_vals['command'] == 'stop':
                    self.log.info("TRIGGERED DUMP >> Stopped")
                    self.update_stats({'last_command' : 'stop',
                                       'status'       : 'stopped'})
                    if not LWA352_DISK_NOOP:
                        del udt
                        os.close(ofile)
                    ofile = None
                    break
                if self.command_vals['command'] == 'abort':
                    self.log.info("TRIGGERED DUMP >> Aborted")
                    self.update_stats({'last_command' : 'abort',
                                       'status'       : 'aborted'})
                    if not LWA352_DISK_NOOP:
                        del udt
                        os.close(ofile)
                    ofile = None
                    break
                    
            # Try closing if a file is still around
            if ofile is not None:
                self.log.info("TRIGGERED DUMP >> Stopped unexpectedly")
                self.update_stats({'status'       : 'stream end'})
                if not LWA352_DISK_NOOP:
                    del udt
                    os.close(ofile)
                    
            self.perf_proclog.update({'acquire_time': acquire_time, 
                                      'reserve_time': -1, 
                                      'process_time': process_time,
                                      'gbps': 8*bytes_rpted / process_time / 1e9})
            
            stop_time = time.time()
            elapsed = stop_time - start_time
            gbytesps = total_bytes / 1e9 / elapsed
            self.log.info("TRIGGERED DUMP >> Complete (Wrote %.2f GBytes in %.2f s (%.2f GB/s)" % (total_bytes/1e9, elapsed, gbytesps))
