from .block_control_base import BlockControl
import numpy as np

class TriggeredDumpControl(BlockControl):
    def trigger(self, ntime_per_file=None, nfile=None, dump_path=None):
       self._send_command(command='trigger', nfile=nfile, ntime_per_file=ntime_per_file, dump_path=dump_path)
    def abort(self):
       self._send_command(command='abort')
    def stop(self):
       self._send_command(command='stop')
