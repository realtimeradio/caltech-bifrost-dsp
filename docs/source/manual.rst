========================
LWA-352
X-Engine Software Manual
========================

:Author: Jack Hickish

.. role:: math(raw)
   :format: html latex
..

Introduction
============

The LWA-352 X-Engine processing system performs correlation,
beamforming, and triggered voltage recording for the LWA-352 array. This
document outlines the hardware (Section [sec:hardware]) and software
(Section [sec:software]) which makes up the X-Engine, and details the
user control interface (Section [sec:api]).

Hardware
========

The LWA-352 X-engine system comprises 9 1U dual-socket Silicon Mechanics
*Rackform R353.v7* servers, each hosting a pair of Nvidia GPUs and solid
state memory buffers. Hardware specifications are given in
Table [tab:hardware].

.. list-table::
  :widths: 30 30 100
  :header-rows: 1
  :align: left

  * - Hardware
    - Model
    - Notes

  * - Server
    - Supermicro 1029GQ-TRT
    - Re-branded as Silicon Mechanics Rackform R353.v7

  * - Motherboard
    - Supermicro X11 DCQ
    - 

  * - CPU
    - dual Intel Xeon Scalable Silver 4210R
    - 2.4 GHz, 10 core, 100W TDP

  * - RAM
    - 768 GB PC4-23400
    - 12 x 64 GB; 2933 MHz DDR4; ECC RDIMM

  * - NIC
    - Mellanox MCX515A-GCAT
    - ConnectX-5 EN MCX515A-GCAT (1x QSFP28); PCIe 3.0x16

  * - NVMe Controllers
    - Asus Hyper M.2 X16 Card V2
    - 2 cards per server

  * - NVMe Memory
    - 8TB Samsung 970 Evo Plus
    - 8 x 1 TB

  * - GPU
    - Nvidia RTX 2080Ti
    - 2 cards per server

Pipeline
========

The pipeline is launched using the ``lwa352-pipeline.py`` script.
This has the following options:

.. program-output:: lwa352-pipeline.py -h


Software
========

The LWA-352 pipeline comprises ?? independent processes, briefly
described below.

#. ``capture``: Receive F-engine packets and correctly arrange in
   buffers for downstream processing. Monitor and record missing packets
   and network performance statistics.

#. ``gpucopy``: Transfer blocks of data from CPU to GPU, for
   high-performance computation.

#. ``corr``: Correlate data using the ``xGPU`` library.

#. ``corrsubsel``: Down-select and transmit a sub-set of the complete
   visibility matrix.

#. **beamform**: Form multiple voltage beams.

#. **beamform\_vlbi**: Package and transmit a single voltage beam fof
   VLBI purposes.

#. **beamform\_vacc**: Form integrated power-spectra for multiple beams,
   and output as UDP streams.

High-Level Parameters
---------------------

-  ``GSIZE``: “Gulp size” – the number of samples processed per batch by
   a processing block.

Bifrost Block Description
-------------------------

The bifrost pipelining framework divides streams of data into
*sequences*, each of which has a header describing the stream. Different
processing blocks act to perform operations on streams, and may modify
sequences and their headers.

Here we summarize the bifrost blocks in the LWA-352 pipeline, and the
headers they require (for input sequences) and provide (for output
sequences).

``capture``
~~~~~~~~~~~

Functionality
^^^^^^^^^^^^^

This block receives data from a high-speed network, and writes it to a
bifrost memory buffer.

New Sequence Condition
^^^^^^^^^^^^^^^^^^^^^^

This block starts a new sequence each time the incoming packet stream
timestamps change in an unexpected way. I.e. if a large block of
timestamps are missed, a new sequence will be started. Or, if the
incoming timestamps decrease (because, for example) the upstream FPGA
transmitters have been re-synchronized, a new sequence is started.

Input Header Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^

This block is a bifrost source, and thus has no input header
requirements.

Output Headers
^^^^^^^^^^^^^^

| This block outputs the following headers:

| c c c X Field & Format & Units & Description
| time\_tag & int & - & Arbirary integer, incremented with each new
  sequence.
| sync\_time & int & UNIX seconds & Synchronization time (corresponding
  to spectrum sequence number 0)
| seq0 & int & - & Spectra number for the first sample in this sequence
| chan0 & int & - & Channel index of the first channel in this sequence
| nchan & int & - & Number of channels in the sequence
| fs\_hz & double & Hz & Sampling frequency of ADCs
| sfreq & double & Hz & Center frequency of first channel in the
  sequence
| bw\_hz & int & Hz & Bandwidth of the sequence
| nstand & int & - & Number of stands (antennas) in the sequence
| npol & int & - & Number of polarizations per stand in the sequence
| complex & bool & - & True if the data are complex, False otherwise
| nbit & int & - & Number of bits per sample (or per real/imag part if
  the samples are complex)
| input\_to\_ant & list of ints & - & List of input to stand/pol
  mappings with dimensions
  :math:`[nstand \times npol, 2]`. E.g. if entry
  :math:`N` of this list has value :math:`[S, P]` then the
  :math:`N^{{th}}` correlator input is stand :math:`S`,
  polarization :math:`P`.
| ant\_to\_input & list of ints & - & List of stand/pol to correlator
  input number mappings with dimensions
  :math:`[{nstand}, {npol}]`. E.g. if entry :math:`[S,P]`
  of this list has value :math:`N` then stand :math:`S`, polarization
  :math:`P` of the array is the :math:`N^{{th}}` correlator input

``gpucopy``
~~~~~~~~~~~

Functionality
^^^^^^^^^^^^^

This block copies data from CPU to GPU memory.

New Sequence Condition
^^^^^^^^^^^^^^^^^^^^^^

This block has no new sequence condition. It will output a new sequence
only when the upstream sequence changes.

Input Header Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^

This block has no input header requirements.

Output Headers
^^^^^^^^^^^^^^

This block copies headers downstream, adding none of its own.

``corr``
~~~~~~~~

Functionality
^^^^^^^^^^^^^

This block performs correlation of voltage data, delivering short-term
(:math:`~1`\ ms) integrated visibilities.

New Sequence Condition
^^^^^^^^^^^^^^^^^^^^^^

This block starts a new sequence each time a new integration
configuration is loaded or the upstream sequence changes.

Input Header Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^

| This block requires the following headers from upstream:

| c c c X Field & Format & Units & Description
| seq0 & int & - & Spectra number for the first sample in the input
  sequence

Output Headers
^^^^^^^^^^^^^^

| This block copies headers from upstream with the following
  modifications:

| c c c X Field & Format & Units & Description
| seq0 & int & - & Spectra number for the *first* sample in the
  integrated output
| acc\_len & int & - & Number of spectra integrated into each output
  sample by this block
| ant\_to\_input & list of ints & - & This header is removed from the
  sequence
| input\_to\_ant & list of ints & - & This header is removed from the
  sequence

``corracc``
~~~~~~~~~~~

Functionality
^^^^^^^^^^^^^

This block further integrates correlation data to deliver long-term
visibility products (:math:`>1s`).

New Sequence Condition
^^^^^^^^^^^^^^^^^^^^^^

This block starts a new sequence each time a new integration
configuration is loaded or the upstream sequence changes.

Input Header Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^

| This block requires the following headers from upstream:

| c c c X Field & Format & Units & Description
| seq0 & int & - & Spectra number for the first sample in the input
  sequence
| acc\_len & int & - & Number of spectra integrated into each output
  sample by the upstream processing

Output Headers
^^^^^^^^^^^^^^

| This block copies headers from upstream with the following
  modifications:

| c c c X Field & Format & Units & Description
| seq0 & int & - & Spectra number for the *first* sample in the
  integrated output
| acc\_len & int & - & Total number of spectra integrated into each
  output sample by this block, incorporating any upstream processing
| upstream\_acc\_len & int & - & Number of spectra integrated by
  upstream processing

``corrsubsel``
~~~~~~~~~~~~~~

Functionality
^^^^^^^^^^^^^

This block selects individual visibilities from the short-term
short-term (:math:`~1`\ ms) correlator integrations and averages them in
frequency.

New Sequence Condition
^^^^^^^^^^^^^^^^^^^^^^

This block starts a new sequence each time a new baseline selection is
loaded or if the upstream sequence changes.

Input Header Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^

| This block requires the following headers from upstream:

| c c c X Field & Format & Units & Description
| seq0 & int & - & Spectra number for the first sample in the input
  sequence
| acc\_len & int & - & Number of spectra accumulated per input data
  sample
| nchan & int & - & Number of channels in the input sequence
| bw\_hz & double & Hz & Bandwith of the input sequence
| sfreq & double & Hz & Center frequency of first channel in the input
  sequence

Output Headers
^^^^^^^^^^^^^^

| This block copies headers from upstream with the following
  modifications:

| c c c X Field & Format & Units & Description
| seq0 & int & - & Spectra number for the first sample in the output
  sequence
| nchan & int & - & Number of frequency channels in the output data
  stream, after any integration performed by this block
| nvis & int & - & Number of visibilities in the output data stream
| baselines & list of ints & - & A list of output stand/pols, with
  dimensions :math:`[{nvis}, 2, 2]`. E.g. if entry :math:`[V]` of
  this list has value :math:`[[N_0, P_0], [N_1, P_1]]` then the
  :math:`V^{{th}}` entry in the output data array is the
  correlation of stand :math:`N_0`, polarization :math:`P_0` with stand
  :math:`N_1`, polarization :math:`P_1`
| bw\_hz & double & Hz & Bandwith of the output sequence, after
  averaging
| sfreq & double & Hz & Center frequency of first channel in the output
  sequence, after averaging
