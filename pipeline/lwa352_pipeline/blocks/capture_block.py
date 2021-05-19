from bifrost.proclog import ProcLog
import bifrost.affinity as cpu_affinity
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture, UDPVerbsCapture

import time
import ujson as json
import threading
import ctypes
import numpy as np

class Capture(object):
    """
    **Functionality**

    This block receives UDP/IP data from an Ethernet network and writes
    it to a bifrost memory buffer.

    **New Sequence Condition**

    This block starts a new sequence each time the incoming packet
    stream timestamp changes in an unexpected way. For example, if a large
    block of timestamps are missed a news sequence will be started. Or, if
    the incoming timestamps decrease (which might happen if the upstream
    transmitters are reset) a new sequence is started.

    **Input Header Requirements**

    This block is a bifrost source, and thus has no input header
    requirements.

    **Output Headers**
    
    Output header fields are as follows:

    .. table::
        :widths: 25 10 10 55

        +------------------+------------+--------------+--------------------------------------------------------------+
        | Field            | Format     | Units        | Description                                                  |
        +==================+============+==============+==============================================================+
        | ``time_tag``     | int        |              | Arbirary integer, incremented with each new sequence.        |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``sync_time``    | int        | UNIX seconds | Synchronization time (corresponding to spectrum sequence     |
        |                  |            |              | number 0)                                                    |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``seq0``         | int        |              | Spectra number for the first sample in this sequence         |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``chan0``        | int        |              | Channel index of the first channel in this sequence          |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``nchan``        | int        |              | Number of channels in the sequence                           |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``system_nchan`` | int        |              | The total number of channels in the system (i.e., the number |
        |                  |            |              | of channels across all pipelines)                            |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``fs_hz``        | double     | Hz           | Sampling frequency of ADCs                                   |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``sfreq``        | double     | Hz           | Center frequency of first channel in the sequence            |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``bw_hz``        | int        | Hz           | Bandwidth of the sequence                                    |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``nstand``       | int        |              | Number of stands (antennas) in the sequence                  |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``npol``         | int        |              | Number of polarizations per stand in the sequence            |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``complex``      | bool       |              | True if the data are complex, False otherwise                |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``nbit``         | int        |              | Number of bits per sample (or per real/imag part if the      |
        |                  |            |              | samples are complex)                                         |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``input_to_ant`` | list[int]  |              | List of input to stand/pol mappings with dimensions          |
        |                  |            |              | ``[nstand x npol, 2]``. E.g. if entry ``N`` of this list has |
        |                  |            |              | value ``[S, P]`` then the ``N``-th correlator input is stand |
        |                  |            |              | ``S``, polarization ``P``.                                   |
        +------------------+------------+--------------+--------------------------------------------------------------+
        | ``ant_to_input`` | list[ints] |              | List of stand/pol to correlator input number mappings with   |
        |                  |            |              | dimensions ``[nstand, npol]``. E.g. if entry ``[S,P]`` of    |
        |                  |            |              | this list has value ``N`` then stand ``S``, polarization     |
        |                  |            |              | ``P`` of the array is the ``N``-th correlator input          |
        +------------------+------------+--------------+--------------------------------------------------------------+

    **Data Buffers**
    
    *Input data buffer*: None

    *Output data buffer*: Complex 4-bit data with dimensions (slowest to fastest)
    ``Time x Freq x Stand x Polarization``

    **Instantiation**

    :param log: Logging object to which runtime messages should be
        emitted.
    :type log: logging.Logger

    :param fs_hz: Sampling frequency, in Hz, of the upstream ADCs
    :type fs_hz: int

    :param chan_bw_hz: Bandwidth of a single frequency channel in Hz.
    :type chan_bw_hz: float

    :param nstand: Number of stands in the array.
    :type nstand: int

    :param npol: Number of polarizations per antenna stand.
    :type npol: int

    :param input_to_ant: An map of correlator input to station /
        polarization. Provided as an ``[nstand x npol, 2]`` array such
        that if ``input_to_ant[i] == [S,P]`` then the i-th correlator
        input is stand ``S``, polarization ``P``.

    *Keyword Arguments*

    :param fmt: The string identifier of the packet format to be
        received. E.g "snap2".
    :type fmt: string

    :param sock: Input UDP socket on which to receive.
    :type sock: bifrost.udp_socket.UDPSocket

    :param ring: bifrost output data ring
    :type ring: bifrost.ring.Ring

    :param core: CPU core to which this block should be bound.
    :type core: int

    :param nsrc: Number of packet sources. This might mean the number of
        boards transmitting packets, or, in the case that it takes multiple
        packets from each board to send a complete set of data, this could
        be a multiple of the number of source boards.
    :type nsrc: int

    :param src0: The first source to transmit to this block.
    :type src0: int

    :param system_nchan: The total number of channels in the complete,
        multi-pipeline system. This is only used to set sequence headers
        for downstream packet header generators which require this information.
    :type system_nchan: int

    :param max_payload_size: The maximum payload size, in bytes, of the
        UDP packets to be received.
    :type max_payload_size: int

    :param buffer_ntime: The number of time samples to be buffered into the
        output data ring buffer before it is marked full.
    :type buffer_ntime: int

    :param utc_start: ?The time at which the block should begin
        receiving. Set to datetime.datetime.now() to start immediately.
    :type utc_start: datetime.datetime

    :param ibverbs: Boolean parameter which, if true, will cause this
        block to use an Infiniband Verbs packet receiver. If false, or not
        provided,  a standard UDP socket will be used.
    :type ibverbs: Bool
    
    """

    def __init__(self, log, fs_hz=196000000, chan_bw_hz=23925.78125,
                     input_to_ant=None, nstand=352, npol=2, system_nchan=182*16,
                     *args, **kwargs):
        self.log    = log
        self.fs_hz  = fs_hz # sampling frequency in Hz
        self.chan_bw_hz = chan_bw_hz # Channel bandwidth in Hz
        self.args   = args
        self.kwargs = kwargs
        self.core = self.kwargs.get('core', 0)
        self.utc_start = self.kwargs['utc_start']
        self.nstand = nstand
        self.npol = npol
        self.system_nchan = system_nchan
        if 'ibverbs' in self.kwargs:
            if self.kwargs['ibverbs']:
                self.log.info("Using IBVERBs")
                self.CaptureClass = UDPVerbsCapture
            else:
                self.log.info("Using Vanilla UDP Capture")
                self.CaptureClass = UDPCapture
            del self.kwargs['ibverbs']
        else:
            self.CaptureClass = UDPCapture

        del self.kwargs['utc_start']
        # Add gulp size = slot_ntime requirement which is special
        # for the LWA352 receiver
        #self.kwargs['slot_ntime'] = kwargs['buffer_ntime']
        self.shutdown_event = threading.Event()

        # make an array ninputs-elements long with [station, pol] IDs.
        # e.g. if input_to_ant[12] = [27, 1], then the 13th input is stand 27, pol 1
        if input_to_ant is not None:
            self.input_to_ant = input_to_ant
        else:
            self.input_to_ant = np.zeros([nstand*npol, 2], dtype=np.int32)
            for s in range(nstand):
                for p in range(npol):
                    self.input_to_ant[npol*s + p] = [s, p]

        self.ant_to_input = np.zeros([nstand, npol], dtype=np.int32)
        for i, inp in enumerate(self.input_to_ant):
            stand = inp[0]
            pol = inp[1]
            self.ant_to_input[stand, pol] = i
        self.time_tag = 0
           
        ## HACK TESTING
        #self.seq_callback = None
    def shutdown(self):
        """
        Shutdown this block.
        """
        self.shutdown_event.set()
    def seq_callback(self, seq0, chan0, nchan, nsrc,
                     time_tag_ptr, hdr_ptr, hdr_size_ptr):
        """
        Callback function invoked each time the underlying C receiver
starts a new sequence.

        :param seq0: The sequence value of the first sample in this
            sequence.
        :type seq0: int
        :param chan0: The ID of the first channel in this sequence.
        :type chan0: int
        :param nchan: The number of channels in this sequence.
        :type nchan: int
        :param system_nchan: The number of channels in the complete,
            multi-pipeline system.
        :type system_nchan: int
        :param nsrc: The number of distinct sources in this sequence.
        :type nsrc: int
        :param time_tag_ptr: A pointer to the underlying time tag used
            by the C receiver. Set this time tag using: ``time_tag_ptr[0] =
            time_tag``
        :param hdr_size_ptr: A pointer to the header size variable used
            by the C receiver.
            Set this with ``hdr_size_ptr[0] = header_size``
        :param hdr_ptr: A pointer to the underlying sequence header used
            by the C receiver.
            Set this by creating a string, and assigning it. Eg:

            ``header_string = json.dumps(header_dictionary).encode()
            header_buf = ctypes.create_string_buffer(header_string)
            hdr_ptr[0] = ctypes.cast(header_buf, ctypes.c_void_p)
            hdr_size_ptr[0] = len(header_string)``

        """
        #t0 = time.time()
        self.time_tag += 1
        sync_time = time_tag_ptr[0]
        #print("++++++++++++++++ seq0     =", seq0)
        #print("                 time_tag =", self.time_tag)
        #print("                 sync_time =", time.ctime(sync_time))
        #self.log.info("Capture >> New sequence at %s" % time.ctime())
        time_tag_ptr[0] = self.time_tag
        nchan = nchan * (nsrc * 32 // self.nstand)
        hdr = {'time_tag': self.time_tag,
               'sync_time': sync_time,
               'seq0':     seq0, 
               'chan0':    chan0,
               'nchan':    nchan,
               'system_nchan':    self.system_nchan,
               'fs_hz':    self.fs_hz,
               'sfreq':    chan0*self.chan_bw_hz,
               'bw_hz':    nchan*self.chan_bw_hz,
               'nstand':   self.nstand,
               'input_to_ant': self.input_to_ant.tolist(),
               'ant_to_input': self.ant_to_input.tolist(),
               #'stand0':   src0*16, # TODO: Pass src0 to the callback too(?)
               'npol':     self.npol,
               'complex':  True,
               'nbit':     4}
        #if self.input_to_ant.shape != (nstand, npol):
        #    self.log.error("Input order shape %s does not match data stream (%d, %d)" %
        #                    (self.input_to_ant.shape, nstand, npol))

        hdr_str = json.dumps(hdr).encode()
        #hdr_str = b'\x00\x00\x00\x00'
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        self.header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(self.header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        #t1 = time.time()
        #print(t1-t0)
        return 0
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_snap2(self.seq_callback)
        with self.CaptureClass(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
                #print status
        del capture
