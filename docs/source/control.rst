Control Interface
=================

Control and monitoring of the X-Engine pipeline is carried out through
the passing of JSON-encoded messages through an ``etcd``\  [1]_
key-value store. Each processing block in the LWA system has a unique
identifier which defines a key to which runtime status is published and
a key which should be monitored for command messages.

The unique key of a processing block is derived from the ``blockname``
of the module within the pipeline, the ``hostname`` of the server on
which a pipeline is running, the pipeline id - ``pipelineid`` - of this
pipeline, and the index of the block - ``blockid`` - which can disambiguate
multiple blocks of the same type which might be present in a pipeline.

The key to which status information is published is:

``/mon/corr/xeng/<hostname>/pipeline/<pipelineid>/<blockname>/<blockid>/status``

The key to which users should write commands is

``/cmd/corr/xeng/<hostname>/pipeline/<pid>/<blockname>/<blockid>/ctrl``

The status key contains a JSON-encoded dictionary of status information
reported by the pipeline. The command key allows a user to send runtime
configuration to a pipeline block, also in JSON dictionary format.

Some fields in these status and command messages are common amongst blocks,
while others are block-specific.

Users can either interact directly with the etcd keys, and perform encoding
and decoding as appropriate, or can use the Pythonic control interface
provided in the ``lwa352-pipeline-control`` library.

The following information about the interface is auto-generated from the
docstrings in this package.

Common Fields
-------------

Control
~~~~~~~~

There are no control fields which are not block-specific.

Status
~~~~~~

TODO.

Block-Specific Fields
---------------------

Correlator Full Visibility Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full correlator visibility output packet stream is controlled by the
``CorrOutputFull`` block, and has the following interface.

Control
+++++++

.. autoclass:: lwa352_pipeline_control.blocks.corr_output_full_control.CorrOutputFull
  :no-show-inheritance:
  :no-members: get_status

Status
++++++

.. autoclass:: lwa352_pipeline_control.blocks.corr_output_full_control.CorrOutputFull
  :noindex:
  :no-show-inheritance:
  :members: get_status

.. [1]
   See `etcd.io <etcd.io>`__
