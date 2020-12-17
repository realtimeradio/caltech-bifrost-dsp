class BlockControl():
    """
    A class for controlling bifrost correlator blocks via
    an etcd interface.

    :param log: Logging object to which log messages should be sent
    :type log: logging.Logger

    :param corr_interface: An ``EtcdCorrControl`` interface instance to the etcd
        host which controls the correlator.
    :type corr_interface: etcd_corr_control.EtcdCorrControl

    :param host: The hostname of the server which this ``BlockControl``
        object should control
    :type host: string

    :param pipeline_id: The 0-indexed pipeline_id on ``host`` which this
        ``BlockControl`` instance should control. Useful if the server
        is running more than one pipeline
    :type pipeline_id: int

    :param name: The name of the pipeline block which this ``BlockControl``
        instance should control. By default this is the name of the control
        class. The name given (or auto-assigned) here should match the
        name of the pipeline block, which is generally the same as the
        block class name
    :type name: string

    """

    _instance_count = -1

    @classmethod
    def _get_instance_id(cls):
        """
        Get an auto-incrementing ID number for a Block of a particular
        class type.
        :param cls: ``Block`` class, e.g. ``BeamformOutputBlock``
        :return: Number of instances of this class currently constructed.
        """
        cls._instance_count += 1
        return cls._instance_count

    def __init__(self, log, corr_interface, host, pipeline_id=0, name=None):
        self._corr_interface = corr_interface
        self._name = name or type(self).__name__
        self._host = host
        self._pipeline_id = pipeline_id
        self.instance_id = self._get_instance_id()
        self._log = log
    
    def _send_command(self, **kwargs):
        self._corr_interface.send_command(
            self._host,
            self._pipeline_id,
            self._name,
            self.instance_id,
            **kwargs,
        )

    def get_status(self, user_only=True):
        return self._corr_interface.get_status(
                   self._host,
                   self._pipeline_id,
                   self._name,
                   self.instance_id,
                   user_only=user_only,
               )
