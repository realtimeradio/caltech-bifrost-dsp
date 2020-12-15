import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.packet_writer import DiskWriter, HeaderInfo

import time
import simplejson as json
import numpy as np
import os
import mmap
import struct

from .block_base import Block

HEADER_SIZE = 1024*1024

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
            hsize = struct.unpack('<I', fh.read(4))
            hblock_size = struct.unpack('<I', fh.read(4))
            header = json.loads(fh.read(hsize))
            fh.seek(hblock_size)
            data = fh.read() # read to EOF


    """

    def __init__(self, log, iring, ntime_gulp=2500, ntime_per_file=1000000,
                 guarantee=True, core=-1, nbyte_per_time=192*352*2, etcd_client=None, dump_path='/fastdata'):

        super(TriggeredDump, self).__init__(log, iring, None, guarantee, core, etcd_client=etcd_client)
        self.ntime_gulp = ntime_gulp
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nbyte_per_time
        self.command = None
        self.ntime_per_file = ntime_per_file
        self.nfile = 1
        self.dump_path = dump_path
        self.nbyte_per_time = nbyte_per_time

    def _etcd_callback(self, watchresponse):
        v = json.loads(watchresponse.events[0].value)
        self.acquire_control_lock()
        if 'command' in v and isinstance(v['command'], str):
            if v['command'] in ['trigger', 'abort', 'stop']:
                self.command = v['command']
            else:
                self.log.error("TRIGGERED DUMP >> Unknown command %s" % v['command'])
        if 'ntime_per_file' in v and isinstance(v['ntime_per_file'], int):
            self.ntime_per_file = v['ntime_per_file']
        if 'nfile' in v and isinstance(v['nfile'], int):
            self.nfile = v['nfile']
        if 'dump_path' in v and isinstance(v['dump_path'], str):
            if os.path.isdir(v['dump_path']):
                self.dump_path = v['dump_path']
            else:
                self.log.error("TRIGGERED DUMP >> Path %s is not a valid directory" % v['dump_path'])
        self.release_control_lock()

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        # Create a 4kB-aligned 1M buffer to store header data.
        # Needs to be 512-byte aligned to work with O_DIRECT writing.
        # Luckily mmap aligns to memory page size
        hinfo = mmap.mmap(-1, HEADER_SIZE)
        start = False
        last_trigger_time = None
        dump_path = self.dump_path
        ntime_per_file = 0
        nfile = 1
        this_time = 0
        filename = None
        ofile = None
        while not self.iring.writing_ended():
            # parse trigger
            if self.command == 'trigger':
                self.acquire_control_lock()
                ntime_per_file = self.ntime_per_file
                nfile = self.nfile
                dump_path = self.dump_path
                self.command = None
                self.release_control_lock()
                last_trigger_time = time.time()
                filename = os.path.join(dump_path, "lwa-dump-%.2f.tbf" % last_trigger_time)
                self.stats.update({'last_trigger_time' : last_trigger_time,
                                   'last_command' : 'trigger',
                                   'ntime_per_file' : ntime_per_file,
                                   'nfile' : nfile,
                                   'dump_path' : dump_path,
                                   'state' : 'triggering'})
                start = True
                self.update_stats()
            if start:
                # Once we make it out of here, go back to idle
                start = False
                file_num = 0
                file_ndumped = 0
                total_bytes = 0
                start_time = time.time()
                with self.iring.open_earliest_sequence(guarantee=self.guarantee) as iseq:
                    ihdr = json.loads(iseq.header.tostring())
                    for ispan in iseq.read(self.igulp_size):
                        #print("size:", ispan.size)
                        #print("offset:", ispan.offset)
                        if ispan.size < self.igulp_size:
                            continue
                        this_time = ihdr['seq0'] + ispan.offset / self.nbyte_per_time
                        ihdr['seq'] = this_time
                        if ofile is None or file_ndumped >= ntime_per_file:
                            if file_ndumped >= ntime_per_file:
                                # Close file and increment file number
                                os.close(ofile)
                                ofile = None
                                file_num += 1
                            if file_num == nfile:
                                break
                            # If we're here we need to open a new file
                            # Doing the writing from python with directIO -- yikes!
                            # We can only write to such a file with 512-byte aligned
                            # access. Bifrost should guarantee this if it was compiled with
                            # such an option
                            self.stats.update({'status' : 'writing'})
                            self.update_stats()
                            self.log.info("TRIGGERED DUMP >> Opening %s" % (filename + '.%d' % file_num))
                            file_ndumped = 0
                            ofile = os.open(
                                        filename + '.%d' % file_num,
                                        os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_DIRECT | os.O_SYNC,
                                    )
                            header = json.dumps(ihdr).encode()
                            hsize = len(header)
                            hinfo.seek(0)
                            hinfo.write(struct.pack('<2I', hsize, HEADER_SIZE) + json.dumps(ihdr).encode())
                            os.write(ofile, hinfo)
                        # Write the data
                        os.write(ofile, ispan.data)
                        file_ndumped += self.ntime_gulp
                        total_bytes += self.igulp_size
                        if self.command == 'stop':
                            self.log.info("TRIGGERED DUMP >> Stopped")
                            self.stats.update({'last_command' : 'stop',
                                               'status'       : 'stopped'})
                            self.update_stats()
                            break
                        if self.command == 'abort':
                            self.log.info("TRIGGERED DUMP >> Aborted")
                            self.stats.update({'last_command' : 'abort',
                                               'status'       : 'aborted'})
                            self.update_stats()
                            break
                        if ispan.size < self.igulp_size:
                            self.log.info("TRIGGERED DUMP >> Stopped because stream ended")
                            self.stats.update({'status'       : 'stream end'})
                            self.update_stats()
                            # ignore final gulp and stop
                            break
                    # Try closing
                    if ofile is not None:
                        os.close(ofile)
                        ofile = None
                    stop_time = time.time()
                    elapsed = stop_time - start_time
                    gbytesps = total_bytes / 1e9 / elapsed
                    self.log.info("TRIGGERED DUMP >> Complete (Wrote %.2f GBytes in %.2f s (%.2f GB/s)" % (total_bytes/1e9, elapsed, gbytesps))
                    self.stats.update({'status' : 'finished'})
                    self.update_stats()
