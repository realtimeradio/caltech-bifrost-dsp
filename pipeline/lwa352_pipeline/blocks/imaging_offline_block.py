from bifrost.pipeline import TransformBlock
import numpy as np

class FrequencySelectBlock(TransformBlock):
    """
    A custom TransformBlock used for selecting a specific range of frequencies 
    from the input data in the bifrost pipeline. This block updates the header 
    to reflect the new frequency range and modifies the data accordingly.

    This selection is necessary due to the fact that GPUs have limited memory 
    capacity. When the input data is upchannelized, it becomes too large to fit 
    into the available GPU memory.

    Parameters:
        iring: Input ring buffer from which data is read.
        start_freq (int): The starting index of the frequency range to be selected.
        end_freq (int): The ending index of the frequency range to be selected.

    Attributes:
        start_freq_idx (int): Index of the first frequency to be selected.
        end_freq_idx (int): Index of the last frequency to be selected.
    """
    def __init__(self, iring, start_freq, end_freq, *args, **kwargs):
        super(FrequencySelectBlock, self).__init__(iring, *args, **kwargs)
        self.start_freq_idx = start_freq
        self.end_freq_idx = end_freq

    def on_sequence(self, iseq):
        """
        Processes the header of the incoming sequence. Adjusts the header information 
        to reflect the new selected frequency range, including updating the number of 
        channels, the frequency scales, and other relevant metadata.

        Parameters:
            iseq: The input sequence from the bifrost pipeline, containing the header 
                  to be modified.

        Returns:
            ohdr: Modified header with updated frequency range information.
        """
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
        """
        Processes each incoming data span by selecting the frequency range specified 
        by `start_freq_idx` and `end_freq_idx`.

        Parameters:
            ispan: The input span from the bifrost pipeline, containing the data to be processed.
            ospan: The output span where the selected frequency data is written.
        """
        in_data = ispan.data.view(np.ndarray)

        selected_data = in_data[:, self.start_freq_idx:self.end_freq_idx, ...]
        print('selected_data shape', selected_data.shape)
        ospan.data[...] = selected_data
        return 


