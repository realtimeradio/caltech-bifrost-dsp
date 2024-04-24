from bifrost.pipeline import TransformBlock
import numpy as np

class FrequencySelectBlock(TransformBlock):
    def __init__(self, iring, start_freq, end_freq, *args, **kwargs):
        super(FrequencySelectBlock, self).__init__(iring, *args, **kwargs)
        self.start_freq_idx = start_freq
        self.end_freq_idx = end_freq

    def on_sequence(self, iseq):
        ohdr = iseq.header.copy()

        ohdr['system_nchan'] = self.end_freq_idx-self.start_freq_idx
        ohdr['_tensor']['shape'][1]= self.end_freq_idx-self.start_freq_idx
        # TODO: uUpdate the header to reflect the new frequency range if needed


        return ohdr

    def on_data(self, ispan, ospan):
        in_data = ispan.data.view(np.ndarray)

        selected_data = in_data[:, self.start_freq_idx:self.end_freq_idx, ...]
        print('selected_data shape', selected_data.shape)
        ospan.data[...] = selected_data
        return 


