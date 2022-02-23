import time
import numpy as np

from .block_control_base import BlockControl

class BeamformControl(BlockControl):
    nchan = 8192
    fs_hz = 196000000
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

    def update_delays(self, beam_id, delays, amps=None, load_time=None, time_unit='time'):
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

        :param load_time: The time at which provided delays / amplitudes
            should be loaded. If None is provided, updates occur immediately.
        :type load_sample: int

        :param time_unit: The unit at which the load time is specified. If 'time', the
            load time should be provided as a UNIX time. If 'sample', the load time should
            be specified as a sample (spectra) index.
        :type time_unit: str

        """
        if amps is None:
            amps = np.ones_like(delays)
        if load_time is None:
            load_sample = -1
        else:
            if time_unit == 'sample':
                load_sample = load_time
            elif time_unit == 'time':
                load_adc_sample = int(load_time * self.fs_hz)
                load_sample = load_adc_sample // (2*nchan)
            else:
                self._log.error('Only time units "sample" and "time" are understood')
                return
        return self._send_command(
            coeffs = {
                'type': 'beamcoeffs',
                'beam_id': beam_id,
                'data': {'delays': delays.tolist(), 'amps': amps.tolist()},
                'load_sample': load_sample
            }
        )
