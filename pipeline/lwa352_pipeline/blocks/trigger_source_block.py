from bifrost.proclog import ProcLog
import bifrost.ndarray as BFArray
import bifrost.affinity as cpu_affinity

import os
import time
import struct
import ujson as json
import threading
import numpy as np

from bifrost.pipeline import SourceBlock

class TrigBufSourceBlock(SourceBlock):
    def __init__(self, sourcenames, gulp_nframe=1, frame_size=1, *args, **kwargs):
        self.frame_size = frame_size
        super(TrigBufSourceBlock, self).__init__(sourcenames,
                                                 gulp_nframe=gulp_nframe,
                                                 *args, **kwargs)
    def create_reader(self, sourcename):
        return open(sourcename, 'rb')

    def _read_header(self, reader):
        hsize = struct.unpack('<I', reader.read(4))[0]
        hblock_size = struct.unpack('<I', reader.read(4))[0]
        header = json.loads(reader.read(hsize))
        reader.seek(hblock_size) # Go to where data starts
        return header

    def on_sequence(self, reader, sourcename):
        previous_pos = reader.tell()
        hdr = self._read_header(reader)
        tstep_s = hdr['nchan'] / hdr['bw_hz']
        fstep_hz = 1. / tstep_s
        # Add some axis labels to help downstream bifrost magic
        tstart_unix = hdr['seq'] * tstep_s
        hdr['_tensor'] = {
                'dtype':  'ci' + str(hdr['nbit']),
                'shape':  [-1, self.frame_size, hdr['nchan'], hdr['nstand'], hdr['npol']],
                # Note: 'time' (aka block) is the frame axis
                'labels': ['time', 'fine_time', 'freq', 'stand', 'pol'],
                'scales': [(tstart_unix, tstep_s * self.frame_size),
                           (0, tstep_s),
                           (hdr['sfreq'], fstep_hz),
                           None,
                           None],
                'units':  ['s', 's', 'Hz', None, None],
                'gulp_nframe': self.gulp_nframe,
            }
        # Note: This gives 32 bits to the fractional part of a second,
        #         corresponding to ~0.233ns resolution. The whole part
        #         gets at least 31 bits, which will overflow in 2038.
        time_tag  = int(round(tstart_unix * 2**32))
        hdr['time_tag'] = time_tag
        print(hdr)
        return [hdr]

    def on_data(self, reader, ospans):
        ospan = ospans[0]
        odata = ospan.data
        nbyte = reader.readinto(odata)
        print("read %d bytes " % nbyte)
        # Ignore the last chunk of data
        if nbyte < (ospan.frame_nbyte * self.gulp_nframe):
            return [0]
        if nbyte % ospan.frame_nbyte:
            raise IOError("Block data is truncated")
        nframe = nbyte // ospan.frame_nbyte
        return [nframe]



class TriggerReplay(object):
    """
    ** For Functionality see dummy_source_block.py**
    """

    def __init__(self, log, oring, ntime_gulp=2500,
                 core=-1, nchan=192, nstand=352, npol=2, skip_write=False,
                 target_throughput=22.0, testfile=None, pipeline_id=0):
        self.log = log
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.pipeline_id = pipeline_id
        self.core = core
        self.nchan = nchan
        self.npol = npol
        self.nstand = nstand
        self.ninputs = nstand * npol
        self.skip_write = skip_write
        self.target_throughput = target_throughput


        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")

        self.out_proclog.update({'nring': 1, 'ring0': self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.gulp_size = self.ntime_gulp * nchan * nstand * npol * 1  # complex8


        # file containing test data
        if testfile is not None:
            self.testfile_nbytes = os.path.getsize(testfile)
            self.testfile = open(testfile, 'rb')
            hsize = struct.unpack('<I', self.testfile.read(4))[0]
            self.hblock_size = struct.unpack('<I', self.testfile.read(4))[0]
            self.header_base = json.loads(self.testfile.read(hsize))

            pipeline_id_from_file = self.header_base["pipeline_id"]
            nchan_from_file = self.header_base['nchan']
            npol_from_file = self.header_base['npol']
            nstand_from_file = self.header_base['nstand']


            # Check that the values from the file match the expected values
            if pipeline_id_from_file != pipeline_id:
                self.log.error("pipeline_id from file does not match expected value")
            if nchan_from_file != nchan:
                self.log.error("nchan from file does not match expected value")
            if npol_from_file != npol:
                self.log.error("npol from file does not match expected value")
            if nstand_from_file != nstand:
                self.log.error("nstand from file does not match expected value")


            self.testfile.seek(self.hblock_size)
            data = self.testfile.read()


            data = np.frombuffer(data, dtype=np.uint8)
            ntime = len(data) // (self.nchan * self.nstand * self.npol)

            datar = data.reshape([ntime, nchan, nstand, npol])
            self.test_data = BFArray(datar, dtype='u8', space='system')
        else:
            self.log.error("No file was provided")

        # make an array ninputs-elements long with [station, pol] IDs.
        # e.g. if input_to_ant[12] = [27, 1], then the 13th input is stand 27, pol 1
        self.log.info("Making input -> antenna map")
        self.input_to_ant = np.zeros([self.ninputs, 2], dtype=np.int32)
        for s in range(self.nstand):
            for p in range(self.npol):
                self.input_to_ant[self.npol*s + p] = [s, p]

        self.log.info("Making antenna -> input map")
        self.ant_to_input = np.zeros([self.nstand, self.npol], dtype=np.int32)
        for i, inp in enumerate(self.input_to_ant):
            stand = inp[0]
            pol = inp[1]
            self.ant_to_input[stand, pol] = i

        self.log.info("Initializing data block")
        if skip_write:
            self.test_data = BFArray(shape=[ntime_gulp, nchan, nstand, npol], dtype='i8', space='system')
        else:
            self.test_data = BFArray(np.zeros([ntime_gulp, nchan, nstand, npol]),
                                dtype='u8', space='system')
        self.shutdown_event = threading.Event()



    def shutdown(self):
        self.shutdown_event.set()

    def get_testfile_gulp(self, t) -> np.array:
        """
        Get a single gulp from the test file,
        stopping at the end of the file when reached.
        Inputs: t (int) -- time index of gulp. I.e., increment
            by 1 between gulps.
        """
        nbytes = self.gulp_size
        seekloc = self.hblock_size + self.gulp_size * t
        remaining_bytes = self.testfile_nbytes - seekloc

        if remaining_bytes < nbytes:
            self.log.error("Not enough data left in the file for a full gulp")
            return None

        self.testfile.seek(seekloc)
        rawdata = self.testfile.read(nbytes)

        return np.frombuffer(rawdata, dtype=np.uint8).reshape([self.ntime_gulp, self.nchan, self.nstand, self.npol])

    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        time.sleep(0.1)
        self.oring.resize(self.gulp_size, self.gulp_size*4)
        hdr = {}
        hdr.update(self.header_base)
        hdr['input_to_ant'] = self.input_to_ant.tolist()
        hdr['ant_to_input'] = self.ant_to_input.tolist()

        time_tag = 0
        REPORT_PERIOD = 10
        bytes_per_report = REPORT_PERIOD * self.gulp_size
        acquire_time = 0 # this block doesn't have an input ring
        gbps = 0
        extra_delay = 0
        with self.oring.begin_writing() as oring:
            tick = time.time()
            ohdr_str = json.dumps(hdr)
            prev_time = time.time()
            with oring.begin_sequence(time_tag=time_tag, header=ohdr_str) as oseq:
                while not self.shutdown_event.is_set():
                    # Check if end of test file is reached
                    remaining_bytes = self.testfile_nbytes - (self.hblock_size + self.gulp_size * time_tag)
                    if remaining_bytes < self.gulp_size:
                        self.log.info("End of file reached!")
                        break

                    with oseq.reserve(self.gulp_size) as ospan:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        if not self.skip_write:
                            if self.testfile:
                                self.test_data[...] = self.get_testfile_gulp(time_tag)
                            odata = ospan.data_view(shape=self.test_data.shape[:], dtype=self.test_data.dtype)
                            odata[...] = self.test_data[time_tag]
                        time_tag += 1
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': reserve_time, 
                                              'process_time': process_time,
                                              'gbps' : gbps})
                    time.sleep(max(0, extra_delay / REPORT_PERIOD))
                    if time_tag % REPORT_PERIOD == 0:
                        tock = time.time()
                        dt = tock - tick
                        gbps = 8*bytes_per_report / dt / 1e9
                        self.log.info('%d: Sent %d bytes in %.2f seconds (%.2f Gb/s)' % (time_tag // REPORT_PERIOD, bytes_per_report, dt, gbps))
                        target_time = 8*bytes_per_report / self.target_throughput / 1e9
                        extra_delay = target_time - dt + extra_delay
                        tick = tock
        if self.testfile:
            self.log.info("Closing file")
            self.testfile.close()
