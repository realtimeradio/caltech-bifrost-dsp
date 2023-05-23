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

    :param instance_id: The 0-indexed ID of the block, for use in the case
        that a pipeline contains more than one processing module named ``name``.
    :type instance_id: int

    """

    def __init__(self, log, corr_interface, host, pipeline_id=0, name=None,
                 instance_id=0):
        self._corr_interface = corr_interface
        if name:
            self._name = name
        else:
            classname = type(self).__name__
            if classname.endswith('Control'):
                classname = classname[0:-7]
            self._name = classname
        self._host = host
        self._pipeline_id = pipeline_id
        self._instance_id = instance_id
        self._log = log
        self.host = self._host #TODO find/replace
        self.pipeline_id = self._pipeline_id #TODO find/replace
        self.instance_id = self._instance_id #TODO find/replace
    
    def _send_command(self, **kwargs):
        return self._corr_interface.send_command(
            self._host,
            self._pipeline_id,
            self._name,
            self._instance_id,
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
                   self._instance_id,
                   user_only=user_only,
               )

    def get_special_val(self, keyname):
        """
        Get the value associated with a particular keyname
        for this block.

        :param keyname: Name of key to be read
        :type keyname: str

        :return: Value stored under provided key
        """
        return self._corr_interface.read_special(
                   self._host,
                   self._pipeline_id,
                   self._name,
                   self._instance_id,
                   keyname,
               )

    def update_is_pending(self):
        """
        Return True if new parameters are waiting to be loaded. Else, False.
        Also returns False if the block has no command capabilities.

        :return: update_is_pending
        :rtype: Bool
        """
        self._get_status().get('update_is_pending', False)

    def get_curr_sample(self):
        """
        Get the first sample of the last block of data to be processed,
        in spectra counts. Returns -1 if this block is not capable of
        reporting this quantity.

        :return: spectra_count
        :rtype: int
        """
        return self._get_status()['curr_sample']

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
