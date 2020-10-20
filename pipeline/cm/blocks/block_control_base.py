class BlockControl():

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
        self.log = log
    
    def send_command(self, **kwargs):
        self._corr_interface.send_command(self._host, self._pipeline_id, self._name, self.instance_id, **kwargs)

    def get_status(self, user_only=True):
        return self._corr_interface.get_status(self._host, self._pipeline_id, self._name, self.instance_id, user_only=user_only)
