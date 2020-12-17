import time

from .corr_control import Corr

class CorrAcc(Corr):
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
        # can only start on upstream integration boundaries
        upstream_acc_len = status['upstream_acc_len']
        offset = last_count % upstream_acc_len
        rounded_spectra_delay = int(spectra_delay) - (int(spectra_delay) % upstream_acc_len)
        self._log.debug("Spectra delay after rounding to %d is %d" % (upstream_acc_len, rounded_spectra_delay))
        rounded_spectra_delay += offset
        self._log.debug("Spectra delay after applying offset of %d is %d" % (offset, rounded_spectra_delay))
        return rounded_spectra_delay
