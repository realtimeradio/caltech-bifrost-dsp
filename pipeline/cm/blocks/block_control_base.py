class BlockControl():
    def __init__(self, log, corr_interface, host, pipeline_id=0, name=None):
        self._corr_interface = corr_interface
        self._name = name or type(self).__name__
        self._host = host
        self._pipeline_id = pipeline_id
        self.log = log
    
    def send_command(self, **kwargs):
        self._corr_interface.send_command(self._host, self._pipeline_id, self._name, **kwargs)

    def get_status(self, user_only=True):
        return self._corr_interface.get_status(self._host, self._pipeline_id, self._name, user_only=user_only)
