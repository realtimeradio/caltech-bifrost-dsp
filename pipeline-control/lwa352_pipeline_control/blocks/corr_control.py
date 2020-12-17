import time

from .block_control_base import BlockControl

class CorrControl(BlockControl):
    def set_start_time(self, start_time):
        assert isinstance(start_time, int)
        self._send_command(
            start_time=start_time,
        )

    def set_acc_length(self, acc_len):
        assert isinstance(acc_len, int)
        self._send_command(
            acc_len=acc_len,
        )

    def update_is_pending(self):
        return self.get_status()['update_is_pending']

    def get_curr_sample(self):
        return self.get_status()['curr_sample']

    def get_next_allowed_start(self, delay_s):
        status = self.get_status(user_only=False)
        sync_time = status['sync_time']
        self._log.debug("Detected sync time %s" % time.ctime(sync_time))
        spectra_rate = status['bw_hz'] / status['nchan']
        self._log.debug("Computed spectra rate is %d spectra / second" % spectra_rate)
        last_count = status["stats"]["curr_sample"]
        rough_now = sync_time + (last_count / spectra_rate)
        diff = rough_now - time.time()
        if abs(diff) > 10:
            self._log.warning("It looks like the pipeline is %d seconds ahead of the expected spectra number" % diff)
        spectra_delay = (time.time() + delay_s - sync_time) * spectra_rate
        self._log.debug("Spectra delay is %f" % spectra_delay)
        # can only start on xgpu integration boundaries
        xgpu_acc_len = status['stats']['xgpu_acc_len']
        rounded_spectra_delay = int(spectra_delay) - (int(spectra_delay) % xgpu_acc_len)
        self._log.debug("Spectra delay after rounding to %d is %d" % (xgpu_acc_len, rounded_spectra_delay))
        return rounded_spectra_delay

    def triggered_start(self, delay_s):
        self.set_start_time(self.get_next_allowed_start(delay_s))
