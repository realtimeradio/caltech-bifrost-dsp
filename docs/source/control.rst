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

``/mon/corr/x/<hostname>/pipeline/<pipelineid>/<blockname>/<blockid>/status``

The key to which users should write commands is:

``/cmd/corr/x/<hostname>/pipeline/<pid>/<blockname>/<blockid>/ctrl``

On receipt of a command, a processing block will respond on the key:

``/resp/corr/x/<hostname>/pipeline/<pid>/<blockname>/<blockid>/ctrl``

The status key contains a JSON-encoded dictionary of status information
reported by the pipeline. The command key allows a user to send runtime
configuration to a pipeline block, also in JSON dictionary format.

In addition to the pipeline processing blocks, a system control daemon
listens for commands on the key:

``/mon/corr/x/<hostname>``

And responds on key:

``/mon/corr/x/<hostname>``

Some fields in these status and command messages are common amongst blocks,
while others are block-specific.

Users can either interact directly with the etcd keys, and perform encoding
and decoding as appropriate, or can use the Pythonic control interface
provided in the ``lwa352-pipeline-control`` library.

Command Format
--------------

Commands sent to the command key are JSON-encoded dictionaries, and should
have the following fields:

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Field
      - Type
      - Description

    * - id
      - string
      - A unique string associated with this   
        command, used to identify the command's
        response

    * - cmd
      - string
      - Command name. To update control keys of pipeline processing blocks,
        this should be "update". Otherwise, it should reflect the command
        supported by the correlator control daemon.

    * - val
      - dictionary
      - A dictionary containing keys:
          - kwargs (dictionary): Dictionary of pipeline-specific keys to set. These
            should match the control keys definied by individual blocks below.

Allowed values for **``block``** are any of the block names in the processing
pipeline. I.e.:

  - Capture
  - Copy
  - TriggeredDump
  - Corr
  - CorrSubsel
  - CorrOutputPart
  - CorrAcc
  - CorrOutputFull
  - Beamform
  - BeamformVlbiOutput
  - BeamformSumBeams
  - BeamformOutput
  
Additionally, the block name ``xctrl`` may be used to issue high-level system
commands as defined in the ``XengineController`` class.

Allowed values for **``cmd``** are

  - For ``block="xctrl``, and non-private commands of the ``XengineController``
    class
  - For all other blocks, ``cmd`` should be the string ``"update"``.

The **``kwargs``** field of the ``val`` dictionary should contain:

  - For ``block="xctrl``, arguments and values for the ``XengineController``
    method being called.
  - For all other blocks, ``kwargs`` should be a dictionary of
    control keys, and their designed update values, for the
    relevant processing block.

For example, to set the Correlator short term accumulation
length to 4800, a command should be issues with the following
values:

  +--------------+-------------------------------------+
  | Field        | Value                               |
  +==============+=====================================+
  | ``cmd``      | ``"update"``                        |
  +--------------+-------------------------------------+
  | ``val``      | ``{"acc_len": 4800}``:              |
  +--------------+-------------------------------------+

An example of a valid command JSON string, issued with the above parameters
and with ``id="1"`` is:

``'{"cmd": "update", "val": {"block": "delay", "kwargs": {"acc_len": 4800}, "id": "1"}'``

Consult the Pipeline block descriptions for details of the control
keys associated with particular processing blocks.


Response Format
---------------

Every command sent elicits the writing of JSON-encoded dictionary to the
response key.
This dictionary has the following fields:

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Field
      - Type
      - Description

    * - id
      - string
      - A string matching the ``id`` field of
        the command string to which this is a
        response

    * - val
      - dictionary
      - A dictionary containing keys:
          - timestamp (float) The UNIX time when this response was issued.
          - status (string): The string "normal" if the corresponding command was processed without error,
            or "error" if it was not.
          - response (string): The response of the command method, as
            determined by the command.

Processing blocks have the following response strings:

  - "0": Command OK
  - "-1": Command not recognized
  - "-2": Command kwargs had wrong data type
  - "-3": Command invalid

  +-----------------+-----------------------------------------------------------------+
  | Field           | Value                                                           |
  +=================+=================================================================+
  | ``id``          | ``"1"``                                                         |
  +-----------------+-----------------------------------------------------------------+
  | ``val``         | ``{timestamp: 1618060712.8, status: "normal", response: "0"}``  |
  +-----------------+-----------------------------------------------------------------+

or, in JSON-encoded form:

``'{"id": "1", "val": {"timestamp": 1618060712.8, "status": "normal", "response": "0"}}'``

The correlator controller (i.e., ``block="xctrl"``) responds to commands
with the following error strings in the ``response`` field: 

+-----------------------------+----------------------------------------------+
| "JSON decode error"         | Command string could not be JSON-decoded.    |
+=============================+==============================================+
| "Sequence ID not string"    | Sequence ID was not provided in the command  |
|                             | string or decoded to a non-string value.     |
+-----------------------------+----------------------------------------------+
| "Bad command format"        | Received command did not comply with         |
|                             | formatting specifications. E.g. was missing  |
|                             | a required field such as ``block`` or        |
|                             | ``cmd``.                                     |
+-----------------------------+----------------------------------------------+
| "Command invalid"           | Received command doesn't exist in the        |
|                             | ``Snap2Fengine`` API, or is prohibited for   |
|                             | ``etcd`` access.                             |
+-----------------------------+----------------------------------------------+
| "Wrong block"               | ``block`` field of the command decoded to a  |
|                             | block which doesn't exist.                   |
+-----------------------------+----------------------------------------------+
| "Command arguments invalid" | ``kwargs`` key contained missing, or         |
|                             | unexpected keys.                             |
+-----------------------------+----------------------------------------------+
| "Command failed"            | The underlying ``Snap2Fengine`` API call     |
|                             | raised an exception.                         |
+-----------------------------+----------------------------------------------+

Block-Specific Interfaces
-------------------------

Correlator Pipeline Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The correlator ``XengineController`` control class has the following
methods which may be called over etcd using the block name ``xctrl``:

.. autoclass:: lwa352_pipeline_control.lwa352_xeng_etcd_client.XengineController
  :no-show-inheritance:
  :members:

Control Library
---------------

A control library is provided which provides a Python interface to the
underlying etcd API.

Correlator Pipeline Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline_control.lwa352_pipeline_control.Lwa352CorrelatorControl
  :no-show-inheritance:
  :members:

Beamformer Control
~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline_control.blocks.beamform_control.BeamformControl
  :no-show-inheritance:
  :members:
  :exclude-members: get_status


Beamformer Output Control
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline_control.blocks.beamform_output_control.BeamformOutputControl
  :no-show-inheritance:
  :members:
  :exclude-members: get_status

Beamformer VLBI Output Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline_control.blocks.beamform_vlbi_output_control.BeamformVlbiOutputControl
  :no-show-inheritance:
  :members:
  :exclude-members: get_status

Correlator Control
~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline_control.blocks.corr_control.CorrControl
  :no-show-inheritance:
  :members:
  :exclude-members: get_status

Correlator Accumulator Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline_control.blocks.corr_acc_control.CorrAccControl
  :no-show-inheritance:
  :members:
  :exclude-members: get_status

Correlator Baseline Subselect Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline_control.blocks.corr_subsel_control.CorrSubselControl
  :no-show-inheritance:
  :members:
  :exclude-members: get_status

Correlator Partial Visibility Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwa352_pipeline_control.blocks.corr_output_part_control.CorrOutputPartControl
  :no-show-inheritance:
  :members:
  :exclude-members: get_status

Correlator Full Visibility Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full correlator visibility output packet stream is controlled by the
``CorrOutputFull`` block, and has the following interface.

Control
+++++++

.. autoclass:: lwa352_pipeline_control.blocks.corr_output_full_control.CorrOutputFullControl
  :no-show-inheritance:
  :members:
  :exclude-members: get_status

Status
++++++

.. automethod:: lwa352_pipeline_control.blocks.corr_output_full_control.CorrOutputFullControl.get_status

.. [1]
   See `etcd.io <etcd.io>`__
