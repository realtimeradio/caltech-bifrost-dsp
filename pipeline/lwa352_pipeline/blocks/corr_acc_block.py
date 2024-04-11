import bifrost.ndarray as BFArray
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.linalg import LinAlg
from bifrost import map as BFMap
from bifrost.ndarray import copy_array
from bifrost.device import stream_synchronize, set_device as BFSetGPU

import time
import ujson as json
import numpy as np

from .block_base import Block

class CorrAcc(Block):
    """
    **Functionality**

    This block reads data from a GPU-side bifrost ring buffer and accumulates
    it in an internal GPU buffer. The accumulated data are then copied to
    an output ring buffer. 

    **New Sequence Condition**

    This block starts a new sequence each time a new integration
    configuration is loaded or the upstream sequence changes.

    **Input Header Requirements**

    This block requires that the following header fields
    be provided by the upstream data source:

    .. table::
        :widths: 25 10 10 55

        +-------------+--------+-------+------------------------------------------------+
        | Field       | Format | Units | Description                                    |
        +=============+========+=======+================================================+
        | ``seq0``    | int    |       | Spectra number for the first sample in the     |
        |             |        |       | input sequence                                 |
        +-------------+--------+-------+------------------------------------------------+
        | ``acc_len`` | int    |       | Number of spectra integrated into each output  |
        |             |        |       | sample by the upstream processing              |
        +-------------+--------+-------+------------------------------------------------+

    **Output Headers**

    This block passes headers from the upstream block with
    the following modifications:

    .. table::
        :widths: 25 10 10 55

        +----------------------+----------+---------+---------------------------------+
        | Field                | Format   | Units   | Description                     |
        +======================+==========+=========+=================================+
        | ``seq0``             | int      |         | Spectra number for the *first*  |
        |                      |          |         | sample in the integrated output |
        +----------------------+----------+---------+---------------------------------+
        | ``acc_len``          | int      |         | Total number of spectra         |
        |                      |          |         | integrated into each output     |
        |                      |          |         | sample by this block,           |
        |                      |          |         | incorporating any upstream      |
        |                      |          |         | processing                      |
        +----------------------+----------+---------+---------------------------------+
        | ``upstream_acc_len`` | int      |         | Number of spectra already       |
        |                      |          |         | integrated by upstream          |
        |                      |          |         | processing                      |
        +----------------------+----------+---------+---------------------------------+

    **Data Buffers**

    *Input Data Buffer*: A GPU-side bifrost ring buffer of 32+32 bit
    complex integer data. The input buffer is read in gulps of
    ``nchan * (nstand//2+1)*(nstand//4)*npol*npol*4*2`` 32-bit words, which
    is the appropriate size if this block is fed by an upstream ``Corr`` block.
    
    *Note that if the upstream block is ``Corr``, the complexity axis of the input
    buffer is not the fastest changing.*

    *Output Data Buffer*: A bifrost ring buffer of 32+32 bit complex
    integer data of the same ordering and dimensionality as the input buffer.

    The output buffer is written in single accumulation blocks (an integration of
    ``acc_len`` input vectors).

    **Instantiation**

    :param log: Logging object to which runtime messages should be
        emitted.
    :type log: logging.Logger

    :param iring: bifrost input data ring. This should be on the GPU.
    :type iring: bifrost.ring.Ring

    :param oring: bifrost output data ring. This should be on the GPU.
    :type oring: bifrost.ring.Ring

    :param guarantee: If True, read data from the input ring in blocking "guaranteed"
        mode, applying backpressure if necessary to ensure no data are missed by this block.
    :type guarantee: Bool

    :param core: CPU core to which this block should be bound. A value of -1 indicates no binding.
    :type core: int

    :param gpu: GPU device which this block should target. A value of -1 indicates
        no binding
    :type gpu: int

    :param nchan: Number of frequency channels per time sample.
    :type nchan: int

    :param nstand: Number of stands per time sample.
    :type nstand: int

    :param npol: Number of polarizations per stand.
    :type npol: int

    :param acc_len: Accumulation length per output buffer write. This should
        be an integer multiple of any upstream accumulation.
        This parameter can be updated at runtime.
    :type acc_len: int

    :parameter etcd_client: Etcd client object used to facilitate control of this block.
        If ``None``, do not use runtime control.
    :type etcd_client: etcd3.client.Etcd3Client

    :parameter autostartat: The start time at which the correlator should
        automatically being correlating without intervention of the runtime control
        system. Use the value ``-1`` to cause integration to being on the next
        gulp.
    :type autostartat: int

    **Runtime Control and Monitoring**

    This block accepts the following command fields:

    .. table::
        :widths: 25 10 10 55

        +------------------+--------+---------+------------------------------+
        | Field            | Format | Units   | Description                  |
        +==================+========+=========+==============================+
        | ``acc_len``      | int    | samples | Number of samples to         |
        |                  |        |         | accumulate. This should be a |
        |                  |        |         | multiple of any upstream     |
        |                  |        |         | accumulation performed by    |
        |                  |        |         | other blocks. I.e., it       |
        |                  |        |         | should be an integer         |
        |                  |        |         | multiple of a sequences      |
        |                  |        |         | ``acc_len`` header entry.    |
        +------------------+--------+---------+------------------------------+
        | ``start_time``   | int    | samples | The desired first time       |
        |                  |        |         | sample in an accumulation.   |
        |                  |        |         | This should be compatible    |
        |                  |        |         | with the accumulation length |
        |                  |        |         | and start time of upstream   |
        |                  |        |         | blocks. I.e. it should be    |
        |                  |        |         | offset from the input        |
        |                  |        |         | sequence header's ``seq0``   |
        |                  |        |         | value by an integer multiple |
        |                  |        |         | of the input sequence        |
        |                  |        |         | header's ``acc_len`` value   |
        +------------------+--------+---------+------------------------------+

    """

    def __init__(self, log, iring, oring,
                 guarantee=True, core=-1, nchan=192, npol=2, nstand=352, acc_len=24000, gpu=-1, etcd_client=None, autostartat=0):
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        super(CorrAcc, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)
        self.nchan = nchan
        self.npol = npol
        self.nstand = nstand
        self.matlen = nchan * (nstand//2+1)*(nstand//4)*npol*npol*4
        self.gpu = gpu

        if self.gpu != -1:
            BFSetGPU(self.gpu)
        
        self.igulp_size = self.matlen * 8 # complex64
        self.ogulp_size = self.igulp_size
        # integration buffer
        self.accdata = BFArray(shape=(self.igulp_size // 4), dtype='i32', space='cuda')
        self.bfbf = LinAlg()

        self.define_command_key('start_time', type=int, initial_val=autostartat)
        self.define_command_key('acc_len', type=int, initial_val=acc_len)

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        oseq = None
        ospan = None
        start = False
        process_time = 0
        time_tag = 1
        self.update_stats({'state': 'starting'})
        with self.oring.begin_writing() as oring:
            prev_time = time.time()
            self.update_pending = True
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                ohdr = ihdr.copy()
                this_gulp_time = ihdr['seq0']
                upstream_acc_len = ihdr['acc_len']
                ohdr['upstream_acc_len'] = upstream_acc_len
                upstream_start_time = this_gulp_time
                self.sequence_proclog.update(ohdr)
                # If the correlator was running before and a new sequence came
                # try to realign to an appropriate integration boundary and continue
                if start:
                    last_start_time = start_time
                    # number of integrations missed
                    missed_time = (this_gulp_time - last_start_time)
                    missed_accs = missed_time // acc_len
                    # New start time
                    start_time = last_start_time + (missed_accs + 2)*acc_len
                    # Don't start until we get to this time
                    start = False
                    self.log.info("CORRACC >> Recovering start time set to %d. Accumulating %d samples" % (start_time, acc_len))
                    if acc_len % upstream_acc_len != 0:
                        self.log.error("CORRACC >> Requested acc_len %d incompatible with upstream integration %d" % (acc_len, upstream_acc_len))
                    if acc_len != 0 and ((start_time - upstream_start_time) % upstream_acc_len != 0):
                        self.log.error("CORRACC >> Requested start_time %d incompatible with upstream integration %d" % (acc_len, upstream_acc_len))
                    ohdr['acc_len'] = acc_len
                    ohdr['seq0'] = start_time
                for ispan in iseq.read(self.igulp_size):
                    if ispan.size < self.igulp_size:
                        continue # skip last gulp
                    if self.update_pending:
                        self.update_command_vals()
                        acc_len = self.command_vals['acc_len']
                        # Use start_time = -1 as a special condition to start on the next sample
                        if self.command_vals['start_time'] == -1:
                            start_time = this_gulp_time
                        else:
                            start_time = self.command_vals['start_time']
                        start = False
                        self.log.info("CORRACC >> New start time at %d. Accumulation: %d samples" % (start_time, acc_len))
                        self.update_pending = False
                        if acc_len % upstream_acc_len != 0:
                            self.log.error("CORRACC >> Requested acc_len %d incompatible with upstream integration %d" % (acc_len, upstream_acc_len))
                        if acc_len != 0 and ((start_time - upstream_start_time) % upstream_acc_len != 0):
                            self.log.error("CORRACC >> Requested start_time %d incompatible with upstream integration %d" % (acc_len, upstream_acc_len))
                        ohdr['acc_len'] = acc_len
                        ohdr['seq0'] = start_time
                    self.stats.update({'curr_sample': this_gulp_time})
                    self.update_stats()
                    # Use acc_len = 0 as a special stop condition
                    if acc_len == 0:
                        self.update_stats({'state': 'stopped'})
                        if oseq: oseq.end()
                        oseq = None
                        start = False
                        this_gulp_time += upstream_acc_len
                        continue

                    # If we get here, acc_len is != 0, and we are searching for
                    # a new integration boundary

                    # If this is the start time, update the first flag, and compute where the last flag should be
                    if this_gulp_time == start_time:
                        start = True
                        first = start_time
                        last  = first + acc_len - upstream_acc_len
                        # on a new accumulation start, if a current oseq is open, close it, and start afresh
                        if oseq: oseq.end()
                        ohdr_str = json.dumps(ohdr)
                        self.sequence_proclog.update(ohdr)
                        oseq = oring.begin_sequence(time_tag=time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        time_tag += 1
                        self.log.info("CORRACC >> Start time %d reached. Accumulating to %d (upstream accumulation: %d)" % (start_time, last, upstream_acc_len))

                    # If we're still waiting for a start, spin the wheels
                    if not start:
                        self.update_stats({'state': 'waiting'})
                        this_gulp_time += upstream_acc_len
                        continue

                    self.update_stats({'state': 'running'})

                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    self.log.debug("Accumulating correlation")
                    idata = ispan.data_view('i32')
                    if this_gulp_time == first:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        # TODO: surely there are more sensible ways to implement a vacc
                        BFMap("a = b", data={'a': self.accdata, 'b': idata})
                    else:
                        BFMap("a += b", data={'a': self.accdata, 'b': idata})
                    curr_time = time.time()
                    process_time += curr_time - prev_time
                    prev_time = curr_time
                    if this_gulp_time == last:
                        self.log.debug("CORRACC > Last accumulation input")
                        # copy to CPU
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                        odata = ospan.data_view('i32').reshape(self.accdata.shape)
                        copy_array(odata, self.accdata)
                        # Wait for copy to complete before committing span
                        stream_synchronize()
                        ospan.close()
                        ospan = None
                        curr_time = time.time()
                        process_time += curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,})
                        self.update_stats({'last_end_sample': this_gulp_time})
                        process_time = 0
                        # Update integration boundary markers
                        first = last + upstream_acc_len
                        last = first + acc_len - upstream_acc_len
                    # And, update overall time counter
                    this_gulp_time += upstream_acc_len
            # If upstream process stops producing, close things gracefully
            # TODO: why is this necessary? Get exceptions from ospan.__exit__ if not here
            if ospan: ospan.close()
            if oseq: oseq.end()
