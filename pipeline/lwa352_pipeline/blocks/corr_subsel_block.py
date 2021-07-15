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

class CorrSubsel(Block):
    """
    **Functionality**
    
    This block selects individual visibilities from an xGPU buffer,
    averages them in frequency, and rearranges them into a sane format.
    
    **New Sequence Condition**
    
    This block starts a new sequence each time a new baseline selection is
    loaded or if the upstream sequence changes.
    
    **Input Header Requirements**
    
    This block requires the following headers from upstream:
    
    .. table::
        :widths: 25 10 10 55

        +------------+----------+---------+-------------------------------------------------------------+
        | Field      | Format   | Units   | Description                                                 |
        +============+==========+=========+=============================================================+
        | seq0       | int      | -       | Spectra number for the first sample in the input sequence   |
        +------------+----------+---------+-------------------------------------------------------------+
        | acc\_len   | int      | -       | Number of spectra accumulated per input data sample         |
        +------------+----------+---------+-------------------------------------------------------------+
        | nchan      | int      | -       | Number of channels in the input sequence                    |
        +------------+----------+---------+-------------------------------------------------------------+
        | bw\_hz     | double   | Hz      | Bandwith of the input sequence                              |
        +------------+----------+---------+-------------------------------------------------------------+
        | sfreq      | double   | Hz      | Center frequency of first channel in the input sequence     |
        +------------+----------+---------+-------------------------------------------------------------+

    **Output Headers**

    This block copies headers from upstream with the following
    modifications:

    .. table::
        :widths: 25 10 10 55

        +-------------+----------------+---------+----------------------------------------------------+
        | Field       | Format         | Units   | Description                                        |
        +=============+================+=========+====================================================+
        | seq0        | int            | -       | Spectra number for the first sample in the output  |
        |             |                |         | sequence. This may diverge from the input sequence |
        |             |                |         | ``seq0``.                                          |
        +-------------+----------------+---------+----------------------------------------------------+
        | nchan       | int            | -       | Number of frequency channels in the output data    |
        |             |                |         | stream, after any integration performed by this    |
        |             |                |         | block                                              |
        +-------------+----------------+---------+----------------------------------------------------+
        | nvis        | int            | -       | Number of visibilities in the output data stream   |
        +-------------+----------------+---------+----------------------------------------------------+
        | baselines   | list of ints   | -       | A list of output stand/pols, with dimensions       |
        |             |                |         | ``[nvis, 2, 2]``. E.g. if entry :math:`[V]` of     |
        |             |                |         | this list has value ``[[N_0, P_0], [N_1, P_1]]``   |
        |             |                |         | then the ``V``-th entry in the output data array   |
        |             |                |         | is the correlation of stand ``N_0``, polarization  |
        |             |                |         | ``P_0`` with stand ``N_1``, polarization ``P_1``   |
        +-------------+----------------+---------+----------------------------------------------------+
        | bw\_hz      | double         | Hz      | Bandwith of the output sequence, after averaging   |
        +-------------+----------------+---------+----------------------------------------------------+
        | sfreq       | double         | Hz      | Center frequency of first channel in the output    |
        |             |                |         | sequence, after averaging                          |
        +-------------+----------------+---------+----------------------------------------------------+

    **Data Buffers**

    *Input Data Buffer*: A GPU-side bifrost ring buffer of 32+32 bit
    complex integer data. The input buffer is read in gulps of
    ``nchan * (nstand//2+1)*(nstand//4)*npol*npol*4*2`` 32-bit words, which
    is the appropriate size if this block is fed by an upstream ``Corr`` block.
    
    *Note that if the upstream block is ``Corr``, the complexity axis of the input
    buffer is not the fastest changing.*

    *Output Data Buffer*: A bifrost ring buffer of 32+32 bit complex integer data.
    The output buffer may be in GPU or CPU memory, and has dimensions
    ``time x frequency channel x visibility x complexity``.
    The output buffer is written in blocks of ``nchan // nchan_sum x nvis_out[=4704]``
    64-bit words.

    **Instantiation**

    :param log: Logging object to which runtime messages should be
        emitted.
    :type log: logging.Logger

    :param iring: bifrost input data ring. This should be on the GPU.
    :type iring: bifrost.ring.Ring

    :param oring: bifrost output data ring. This may be on the CPU or GPU.
    :type oring: bifrost.ring.Ring

    :param guarantee: If True, read data from the input ring in blocking "guaranteed"
        mode, applying backpressure if necessary to ensure no data are missed by this block.
    :type guarantee: Bool

    :param core: CPU core to which this block should be bound. A value of -1 indicates no binding.
    :type core: int

    :param nchan: Number of frequency channels per time sample.
    :type nchan: int

    :param nstand: Number of stands per time sample.
    :type nstand: int

    :param npol: Number of polarizations per stand.
    :type npol: int

    :param nchan_sum: Number of frequency channels to sum together when generating output.
    :type nchan_sum: int

    :parameter etcd_client: Etcd client object used to facilitate control of this block.
        If ``None``, do not use runtime control.
    :type etcd_client: etcd3.client.Etcd3Client

    :param antpol_to_bl: Map of antenna/polarization visibility intputs to xGPU output indices.
        See optional sequence header entry ``ant_to_bl_id``.
    :type antpol_to_bl: 4D list of int

    :param bl_is_conj: Map of visibility index to conjugation convention. See optional
        sequence header entry ``bl_is_conj``.
    :type bl_is_conj: 4D list of bool

    **Runtime Control and Monitoring**

    .. table::
        :widths: 25 10 10 55

        +------------------+--------+---------+------------------------------+
        | Field            | Format | Units   | Description                  |
        +==================+========+=========+==============================+
        | ``baselines``    | 3D     |         | A list of baselines for      |
        |                  | list   |         | subselection. This field     |
        |                  | of int |         | should be provided as a      |
        |                  |        |         | multidimensional list with   |
        |                  |        |         | dimensions ``[nvis, 2, 2]``. |
        |                  |        |         | The first axis runs over the |
        |                  |        |         | 4704 baselines which may be  |
        |                  |        |         | selected. The second index   |
        |                  |        |         | is 0 for the first           |
        |                  |        |         | (unconjugated) input         |
        |                  |        |         | selected and 1 for the       |
        |                  |        |         | second (conjugated) input    |
        |                  |        |         | selected. The third axis is  |
        |                  |        |         | 0 for stand number, and 1    |
        |                  |        |         | for polarization number.     |
        +------------------+--------+---------+------------------------------+

    *Example*

    To set the baseline subsection to choose:

      - visibility 0: the autocorrelation of antenna 0, polarization 0
      - visibility 1: the cross correlation of antenna 5, polarization 1 with antenna 6, polarization 0

    use:

    ``subsel = [ [[0,0], [0,0]], [[5,1], [6,0]], ... ]``

    Note that the uploaded selection list must always have 4704 entries.

    """

    nvis_out = 48 * 49 * 4 // 2 # 48-stand, dual-pol
    def __init__(self, log, iring, oring, guarantee=True, core=-1, etcd_client=None,
                 nchan=192, npol=2, nstand=352, nchan_sum=4, gpu=-1,
                 antpol_to_bl=None, bl_is_conj=None):

        super(CorrSubsel, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)

        self.nchan_in = nchan
        self.nchan_out = nchan // nchan_sum
        self.nchan_sum = nchan_sum
        self.npol = npol
        self.nstand = nstand
        self.gpu = gpu
        self.matlen = self.nchan_in * (nstand//2+1)*(nstand//4)*npol*npol*4 # xGPU defined

        if self.gpu != -1:
            BFSetGPU(self.gpu)

        self.igulp_size = self.matlen * 8 # complex64

        # Create an array of subselection indices on the GPU, and one on the CPU.
        # The user can update the CPU-side array, and the main processing thread
        # will copy this to the GPU when it changes
        # TODO: nvis_out could be dynamic, but we'd have to reallocate the GPU memory
        # if the size changed. Leave static for now, which is all the requirements call for.
        self._subsel      = BFArray(shape=[self.nvis_out], dtype='i32', space='cuda')
        self._subsel_next = BFArray(shape=[self.nvis_out], dtype='i32', space='cuda_host')
        self._conj      = BFArray(shape=[self.nvis_out], dtype='i32', space='cuda')
        self._conj_next = BFArray(shape=[self.nvis_out], dtype='i32', space='cuda_host')

        self.obuf_gpu = BFArray(shape=[self.nchan_out, self.nvis_out], dtype='ci32', space='cuda')
        self.ogulp_size = self.nchan_out * self.nvis_out * 8
        self.update_stats()
        if antpol_to_bl is not None:
            self._antpol_to_bl = antpol_to_bl
        else:
            self._antpol_to_bl = np.zeros([nstand, npol, nstand, npol])
        if bl_is_conj is not None:
            self._bl_is_conj = bl_is_conj
        else:
            self._bl_is_conj = np.zeros([nstand, npol, nstand, npol])

        # update subselection map to a default initial value of
        # pol 0 autos
        # This can't be called until the bl_is_conj and antpol_to_bl maps have been set above
        subsel = [[[i % nstand,0], [i % nstand,0]] for i in range(self.nvis_out)]

        self.define_command_key('baselines', type=list, initial_val=subsel,
                                condition=lambda x: len(x) == self.nvis_out)
        # Load the subselection indices
        self.update_subsel(subsel)
        
    def update_subsel(self, baselines):
        """
        Update the baseline index list which should be subselected.
        Updates are not applied immediately, but are transferred to the
        GPU at the end of the current data block.
        """
        cpu_affinity.set_core(self.core)
        for v in range(self.nvis_out):
            i0, i1 = baselines[v]
            s0, p0 = i0
            s1, p1 = i1
            # index as S0, S1, P0, P1
            self._subsel_next.data[v] = self._antpol_to_bl[s0, s1, p0, p1]
            self._conj_next.data[v] = self._bl_is_conj[s0, s1, p0, p1]

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        oseq = None
        time_tag = 1
        with self.oring.begin_writing() as oring:
            prev_time = time.time()
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                this_gulp_time = ihdr['seq0']
                acc_len = ihdr['acc_len']
                # Uncomment this if you want to read the map on the fly
                #antpol_to_bl = ihdr['antpol_to_bl']
                ohdr = ihdr.copy()
                #ohdr.pop('antpol_to_bl')
                ohdr['nchan'] = ihdr['nchan'] // self.nchan_sum
                ohdr['nvis'] = self.nvis_out
                chan_width = ihdr['bw_hz'] / ihdr['nchan']
                ohdr['sfreq'] = (ihdr['sfreq'] + ((self.nchan_sum - 1) * chan_width)) / self.nchan_sum
                # On a start of sequence, always grab new subselection
                self.log.info("Updating baseline subselection indices")
                self.update_command_vals()
                self.update_subsel(self.command_vals['baselines'])
                # copy to GPU
                copy_array(self._subsel, self._subsel_next)
                copy_array(self._conj, self._conj_next)
                ohdr['baselines'] = self.command_vals['baselines']
                ohdr_str = json.dumps(ohdr)
                oseq = oring.begin_sequence(time_tag=time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                time_tag += 1
                for ispan in iseq.read(self.igulp_size):
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    self.log.debug("Grabbing subselection")
                    idata = ispan.data_view('ci32').reshape(self.matlen)
                    with oseq.reserve(self.ogulp_size) as ospan:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        rv = _bf.bfXgpuSubSelect(idata.as_BFarray(), self.obuf_gpu.as_BFarray(), self._subsel.as_BFarray(), self._conj.as_BFarray(), self.nchan_sum)
                        if (rv != _bf.BF_STATUS_SUCCESS):
                            self.log.error("xgpuSubSelect returned %d" % rv)
                            raise RuntimeError
                        odata = ospan.data_view(dtype='ci32').reshape([self.nchan_out, self.nvis_out])
                        copy_array(odata, self.obuf_gpu)
                        # Wait for copy to complete before committing span
                        stream_synchronize()
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': reserve_time, 
                                              'process_time': process_time,
                                              'this_sample' : this_gulp_time})
                    # tick the sequence counter to the next integration
                    this_gulp_time += acc_len
                    # If a baseline change is pending start a new sequence
                    # with an updated header
                    if self.update_pending:
                        oseq.end()
                        self.log.info("Updating baseline subselection indices")
                        self.update_command_vals()
                        self.udpate_subsel(self.command_vals['baselines'])
                        copy_array(self._subsel, self._subsel_next)
                        copy_array(self._conj, self._conj_next)
                        ohdr['baselines'] = self.command_vals['baselines']
                        #update time tag based on what has already been processed
                        ohdr['seq0'] = this_gulp_time
                        ohdr_str = json.dumps(ohdr)
                        oseq = oring.begin_sequence(time_tag=time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        time_tag += 1
                # if the iseq ends
                oseq.end()
