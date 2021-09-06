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

The LWA-352 pipeline comprises 13 independent processes, briefly
described below.

#. ``capture``: Receive F-engine packets and correctly arrange in
   buffers for downstream processing. Monitor and record missing packets
   and network performance statistics.

#. ``copy1``: Transfer blocks of data from the capture buffer to a deep
   transient buffer.

#. ``copy2``: Transfer blocks of data from CPU to GPU, for
   high-performance computation.

#. ``triggered_dump``: Process software triggers to copy deep-buffered data
   to disk.

#. ``corr``: Correlate data using the ``xGPU`` library and accumulate for
   short (~100ms) durations.

#. ``corr_subsel``: Down-select a sub-set of the complete visibility matrices.

#. ``corr_output_part``: Output subselected visibilities as UDP/IP streams.

#. ``corr_acc``: Further accumulate correlation output to ~second durations.

#. ``corr_output_full``: Output full, accumulated visibility matrices.

#. ``beamform``: Form multiple voltage beams.

#. ``beamform_vlbi_output``: Package and transmit voltage beam(s) for
   VLBI purposes.

#. ``beamform_sum_beams``: Form integrated power-spectra for multiple beams.

#. ``beamform_output``: Output accumulated power beams.

High-Level Parameters
---------------------

-  ``NBEAM``: Number of dual-polarization beams to form.
-  ``CHAN_PER_PACKET``: Number of frequency channels in an F-engine packet.
   This should be chosen to match the F-engine configuration.
-  ``NPIPELINE``: The total number of pipelines in the multi-server correlator
   system.
-  ``NSNAP``: The number of SNAP2 boards in the total F-engine system.
-  ``GSIZE``: “Gulp size” – the number of samples processed per batch by
   a processing block.
-  ``NETGSIZE``: Number of time samples buffered in a capture buffer block.
-  ``NET_NGULP``: Number of ``NETGSIZE`` buffers comprising a single buffer
   "time slot". Data capture always begins on a time boundary of
   ``NETGSIZE*NET_NGULP``, and downstream processes should begin processing
   on boundaries consistent with this.

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

.. autoclass:: lwa352_pipeline.blocks.capture_block.Capture

``copy``
~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.copy_block.Copy

``triggered_dump``
~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.triggered_dump_block.TriggeredDump

``corr``
~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.corr_block.Corr

``corr_acc``
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.corr_acc_block.CorrAcc

``corr_output_full``
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.corr_output_full_block.CorrOutputFull

``corrsubsel``
~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.corr_subsel_block.CorrSubsel

``corr_output_part``
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.corr_output_part_block.CorrOutputPart

``beamform``
~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.beamform_block.Beamform

``beamform_sum_beams``
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.beamform_sum_beams_block.BeamformSumBeams

``beamform_output``
~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.beamform_output_block.BeamformOutput

``beamform_vlbi_output``
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline.blocks.beamform_vlbi_output_block.BeamformVlbiOutput

``dummy_source``
~~~~~~~~~~~~~~~~

The Dummy Source block is not used in the default LWA pipeline, but can replace the
``Capture`` block for testing purposes.

.. autoclass:: lwa352_pipeline.blocks.dummy_source_block.DummySource
