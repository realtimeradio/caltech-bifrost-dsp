from bifrost.pipeline import SinkBlock
import h5py
import numpy as np
import bifrost as bf

class VisibilitySaveBlock(bf.pipeline.SinkBlock):
    """
    A SinkBlock for saving cross-correlated visibility data into a single HDF5 file.
    The HDF5 file will contain datasets for visibilities, frequencies, and time stamps.
    
    Attributes:
        filename (str): Filename for the HDF5 file.
    """
    def __init__(self, iring, filename, *args, **kwargs):
        super(VisibilitySaveBlock, self).__init__(iring, *args, **kwargs)
        self.filename = filename
        self.h5file = None
        self.dataset_vis = None

    def on_sequence(self, iseq):
        """
        Initializes the HDF5 file and creates datasets for visibilities, frequencies, and times based on header info.
        
        Parameters:
            iseq: The input sequence from the bifrost pipeline, containing header information.
        """
        header = iseq.header
        self.h5file = h5py.File(self.filename, 'w')
         
        # Determine frequencies
        nchan = header['system_nchan']
        sfreq, fstep_hz = header['_tensor']['scales'][1]
        frequencies = sfreq + np.arange(nchan) * fstep_hz

        # Determine times
        tstart, tstep = header['_tensor']['scales'][0]
        times = np.array([tstart], dtype=np.float64)  # Initial time, will expand dynamically
        self.tstep = tstep
        # Create frequency and time datasets
        self.h5file.create_dataset('freq', data=frequencies, dtype=np.float32)
        self.dataset_time = self.h5file.create_dataset('time', data=times, maxshape=(None,), dtype=np.float64)

        # Create dataset for visibilities
        vis_shape = header['_tensor']['shape']
        dtype = np.dtype(np.complex64) if header['_tensor']['dtype'] == 'cf32' else np.dtype(np.complex128)
        self.dataset_vis = self.h5file.create_dataset('vis', shape=(0,)+tuple(vis_shape[1:]), maxshape=(None,)+tuple(vis_shape[1:]), dtype=dtype)

    def on_data(self, ispan):
        """
        Processes and stores incoming data spans into the 'vis', and appends new times to the 'time' dataset.
        
        Parameters:
            ispan: The input span from the bifrost pipeline, containing the data.
        """
        # Read data
        data = bf.asarray(ispan.data)
        vis_data = data.copy('system').view(np.ndarray)
        nframe = vis_data.shape[0]

        # Append data to 'vis' dataset
        self.dataset_vis.resize(self.dataset_vis.shape[0] + nframe, axis=0)
        self.dataset_vis[-nframe:] = vis_data

        # Append new times
        last_time = self.dataset_time[-1]
        new_times = last_time + np.arange(1, nframe + 1) * self.tstep
        self.dataset_time.resize(self.dataset_time.shape[0] + nframe, axis=0)
        self.dataset_time[-nframe:] = new_times

    def shutdown(self):
        """
        Ensures that the HDF5 file is properly closed when the block is no longer in use.
        """
        if self.h5file:
            self.h5file.close()

