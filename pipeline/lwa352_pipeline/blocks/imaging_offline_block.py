from bifrost.pipeline import TransformBlock
import numpy as np

class FrequencySelectBlock(TransformBlock):
    def __init__(self, iring, start_freq, end_freq, *args, **kwargs):
        super(FrequencySelectBlock, self).__init__(iring, *args, **kwargs)
        self.start_freq_idx = start_freq
        self.end_freq_idx = end_freq

    def on_sequence(self, iseq):
        ohdr = iseq.header.copy()

        # Calculate new number of channels
        nchan_new = self.end_freq_idx - self.start_freq_idx
        ohdr['system_nchan'] = nchan_new
        ohdr['_tensor']['shape'][1] = nchan_new

        # Original frequency scale
        sfreq, fstep_hz = ohdr['_tensor']['scales'][1]

        # Update starting frequency
        new_sfreq = sfreq + self.start_freq_idx * fstep_hz
        ohdr['_tensor']['scales'][1] = [new_sfreq, fstep_hz]

        # Update 'sfreq' and 'bw_hz' in the header
        ohdr['sfreq'] = new_sfreq
        ohdr['bw_hz'] = nchan_new * fstep_hz

        print('ohdr in freq seelct', ohdr)
        return ohdr
    
    
    def on_data(self, ispan, ospan):
        in_data = ispan.data.view(np.ndarray)

        selected_data = in_data[:, self.start_freq_idx:self.end_freq_idx, ...]
        print('selected_data shape', selected_data.shape)
        ospan.data[...] = selected_data
        return 


