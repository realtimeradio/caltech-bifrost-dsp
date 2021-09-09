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
            where entry ``i`` corresponds to the ``i`` th channel
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
        return self._send_command(
            coeffs = {
                'type': 'calgains',
                'input_id': input_id,
                'beam_id': beam_id,
                'data': gains_real.tolist(),
            }
        )

    def update_delays(self, beam_id, delays, amps=None):
        """
        Update geometric delays for a single beam.

        :param beam_id: Zero-indexed Beam ID for which coefficients are
            begin updated.
        :type beam_id: int

        :param delays: Real-valued delays to load, specified in nanoseconds.
            Should be a numpy array with ``nbeam`` entries,
            where entry ``i`` corresponds to the delay to apply to the ``i`` th
            beamformer input.
        :type delays: numpy.array

        :param amps: Real-valued amplitudes to load.
            Should be a numpy array with ``nbeam`` entries,
            where entry ``i`` corresponds to the real-valued scaling to apply to the ``i`` th
            beamformer input. If None, unity scaling is applied.
        :type delays: numpy.array

        """
        if amps is None:
            amps = np.ones_like(delays)
        return self._send_command(
            coeffs = {
                'type': 'beamcoeffs',
                'beam_id': beam_id,
                'data': {'delays': delays.tolist(), 'amps': amps.tolist()},
            }
        )
