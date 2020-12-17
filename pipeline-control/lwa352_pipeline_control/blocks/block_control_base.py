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
        class, with the suffic 'Control' removed.
        The name given (or auto-assigned) here should match the
        name of the pipeline block.
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
        self._name = name or type(self).__name__.rstrip('Control')
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

    def get_bifrost_status(self, user_only=False):
        """
        Get the stats stored in this block's status key

        :param user_only: If True, only grab the ``stats`` sub-key of the
            status dictionary, which contains block-specific information.
            If False, return the full contents of the ``status`` key,
            much of which is set by the bifrost pipeline engine.
            In this case, ``stats`` are still available as a sub-dictionary
            of the returned status.
        :type user_only: Bool

        :return: Block status dictionary
        :rtype: dict
        """
        return self._corr_interface.get_status(
                   self._host,
                   self._pipeline_id,
                   self._name,
                   self.instance_id,
                   user_only=user_only,
               )

    def _get_status(self):
        """
        Get the block stats (i.e., just the statistics that pipeline blocks
        set with their ``updates_stats()`` method, not everything
        bifrost sets behind the scenes.

        This method mainly exists so that other blocks can call it from their 
        own ``get_status`` method and add their own docstrings for
        auto-generation.

        :return: Block status dictionary
        :rtype: dict
        """

        return self.get_bifrost_status(user_only=True)
