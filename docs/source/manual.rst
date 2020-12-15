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

#. ``copy``: Transfer blocks of data from CPU to GPU, for
   high-performance computation.

#. ``triggered_dump``: Buffer large quantities of time-domain data for triggered
   dump to disk.

#. ``corr``: Correlate data using the ``xGPU`` library.

#. ``corr_output_full``: Output full, accumulated visibility matrices.

#. ``corrsubsel``: Down-select a sub-set of the complete visibility matrices.

#. ``corr_output_part``: Output subselected visibilities as UDP/IP streams.

#. ``beamform``: Form multiple voltage beams.

#. ``beamform_vlbi_output``: Package and transmit multiple voltage beams for
   VLBI purposes.

#. ``beamform\_sum_beams``: Form integrated power-spectra for multiple beams.

#. ``beamform\_output``: Output accumulated power beams.

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

``beamform_vlbi_output``
~~~~~~~~~~~~~~~~~~~~~~~~

``beamform_sum_beams``
~~~~~~~~~~~~~~~~~~~~~~

``beamform_output``
~~~~~~~~~~~~~~~~~~~
