import time
import numpy as np

from .block_control_base import BlockControl

class Beamform(BlockControl):
    def update_coeffs(self):
        gains = np.zeros([32, 704]).tolist()
        delays = np.zeros([32, 704]).tolist()
        self._send_command(
            delays=delays,
            gains=gains,
        )
