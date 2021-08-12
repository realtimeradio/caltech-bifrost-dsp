import time
import numpy as np

from .block_control_base import BlockControl

class BeamformControl(BlockControl):
    def update_calibration_gains(self, beam_id, input_id, gains):
        """
        Update calibration gains for a single beam and input.

        :param beam_id: Zero-indexed Beam ID for which coefficients are
            begin updated.
        :type beam_id: int

        :param input_id: Zero-indexed Input ID for which coefficients are
            begin updated.
        :type input_id: int

        :param gains: Complex-valued gains to load. Should be a numpy
            array with a complex data type and ``nchan`` entries,
            where entry ``i`` corresponds to the ``i``th channel
            being processed by this pipeline.
        :type gains: numpy.array

        """
        # We can't send a numpy array through JSON, so encode
        # as a real-valued list with alternating real/imag entries.
        # This allows the standard JSON messaging scheme to be used,
        # But we could equally use binary strings (which would
        # be much more efficient)
        nchan = gains.shape[0]
        gains_real = np.zeros(2*nchan, dtype=np.float32)
        gains_real[0::2] = gains.real
        gains_real[1::2] = gains.imag
        self._send_command(
            coeffs = {
                'type': 'gains',
                'input_id': input_id,
                'beam_id': beam_id,
                'data': gains_real.tolist(),
            }
        )

    def update_delays(self, beam_id, delays):
        """
        Update geometric delays for a single beam.

        :param beam_id: Zero-indexed Beam ID for which coefficients are
            begin updated.
        :type beam_id: int

        :param delays: Real-valued delays to load, specified in nanoseconds.
            Should be a numpy array with ``nbeam`` entries,
            where entry ``i`` corresponds to the delay to apply to the ``i``th
            beamformer input.
        :type delays: numpy.array

        """
        self._send_command(
            coeffs = {
                'type': 'delays',
                'beam_id': beam_id,
                'data': delays.tolist(),
            }
        )
