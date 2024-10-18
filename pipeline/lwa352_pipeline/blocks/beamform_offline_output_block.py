from bifrost.pipeline import SinkBlock
import h5py
import numpy as np
import bifrost as bf


NUPCHAN = 32
SFREQ = 50148437.5 # in Hz
DEF_CHAN = 23925.78125 # in Hz
STIME_UNIX = 1700085191.14752 #Start time of the data to be processed in unix time, currently hard-coded and can be found in header information
TIME_STEP = 0.001337469387755102 # time step, can be found in header information 

NFREQ = 16*NUPCHAN*96 # Total number of frequency bins across all ranges
NSAMP = int(30*60/TIME_STEP) # Total number of time samples for 30 minutes


class HDF5FullSaveBlock(bf.pipeline.SinkBlock):
    """
    A SinkBlock for saving pipeline data into an HDF5 file with a predefined structure 
    (based on the numbers above, should be modified for a specific observation!).

    This block initializes (or opens) an HDF5 file with structures for storing data,
    including intensity ('I'), frequency ('freq'), and time ('time').
    It handles dynamic data input based on the incoming sequence and data spans.

    Attributes:
        filename (str): Path to the HDF5 file.
        total_nfreq (int): Total number of frequency bins.
        total_nsamples (int): Total number of time samples.
        ra (float): Right ascension of the observation target.
        dec (float): Declination of the observation target.
    """

    def __init__(self, iring, filename, ra, dec, *args, **kwargs):
        super(HDF5FullSaveBlock, self).__init__(iring, *args, **kwargs)
        formatted_ra = "{:.2f}".format(ra)
        formatted_dec = "{:.2f}".format(dec)
        self.filename = "{}_{}_{}.hdf5".format(filename, formatted_ra, formatted_dec)

        self.total_nfreq = NFREQ
        self.total_nsamples = NSAMP 
        self.h5file = None
        self.ra = ra
        self.dec = dec
        self._prepare_file()

    def _prepare_file(self):
        """
        Prepares the HDF5 file for data storage, either by creating a new file or opening an 
        existing one for modification. This method checks for the presence of the initial 
        data structure and invokes the creation of this structure if the file is new.
        """
        # Creating a new file or opening an existing file for modification.
        self.h5file = h5py.File(self.filename, 'a')

        # Check if it's a new file by looking for a specific group/dataset.
        if 'Observation1' not in self.h5file:
            self._create_initial_structure()

    def _create_initial_structure(self):
        """
        Creates the initial structure within the HDF5 file, including the main observation group 
        and necessary datasets for intensity ('I'), frequency ('freq'), and time ('time').
        This setup is executed only once when a new HDF5 file is created.
        """
        # Create the Observation group and its attributes
        obs_group = self.h5file.create_group('Observation1')
        attrs = {
            'ARX_Filter': -1.0,
            'ARX_Gain1': -1.0,
            'ARX_Gain2': -1.0,
            'ARX_GainS': -1.0,
            'Beam': 1,
            'DRX_Gain': -1.0,
            'Dec': self.dec,
            'Dec_Units': 'degrees',
            'Epoch': 2000.0,
            'LFFT': self.total_nfreq,
            'RA': self.ra,
            'RA_Units': 'degrees',
            'RBW': DEF_CHAN / NUPCHAN,
            'RBW_Units': 'Hz',
            'TargetName': '',
            'TrackingMode': 'Unknown',
            'nChan': self.total_nfreq,
            'sampleRate': 196000000.0,
            'sampleRate_Units': 'Hz',
            'tInt_Units': 's'
        }
        for attr, value in attrs.items():
            obs_group.attrs[attr] = value
        
        # Create the Tuning1 subgroup
        tuning_group = obs_group.create_group('Tuning1')
        
        # Initialize the 'I', 'freq', and 'time' datasets with placeholders
        tuning_group.create_dataset('I', shape=(self.total_nsamples, self.total_nfreq), dtype=np.float32, fillvalue=0)
        freq_dataset = tuning_group.create_dataset('freq', shape=(self.total_nfreq,), dtype=np.float32)
        freq_values = np.array([SFREQ + DEF_CHAN / NUPCHAN * i for i in range(NFREQ)], dtype=np.float32)
        freq_dataset[:] = freq_values

        time_dtype = np.dtype([('int', np.int64), ('frac', np.float64)])
        time_dataset = obs_group.create_dataset('time', shape=(self.total_nsamples,), dtype=time_dtype)
        obs_group['time'].attrs['format'] = 'unix'
        obs_group['time'].attrs['scale'] = 'utc'
        
        times = STIME_UNIX + np.arange(NSAMP) * TIME_STEP
        times_int = np.floor(times).astype(np.int64)  # Integer part
        times_frac = (times - times_int).astype(np.float64)  # Fractional part
         
        time_array = np.zeros(NSAMP, dtype=time_dtype)
        time_array['int'] = times_int
        time_array['frac'] = times_frac
        
        time_dataset[:] = time_array

    def on_sequence(self, iseq):
        """
        Handles the start of a new sequence, setting up initial parameters based on sequence header information. 
        This includes calculating start time, frequency step, and determining the frequency range and time index 
        for data insertion.

        This method is called automatically by the pipeline when a new sequence of data starts, ensuring that 
        each data block is correctly aligned with its respective time and frequency.

        Parameters:
            iseq: The sequence object with header information.
        """
        self.tstart_unix, self.tstep_s = iseq.header['_tensor']['scales'][0]
        print('tstart unix and step', self.tstart_unix, self.tstep_s)
        
        sfreq, fstep_hz = iseq.header['_tensor']['scales'][2]
        fine_fstep_hz = iseq.header['_tensor']['scales'][5][1]
        self.nchan = iseq.header['nchan']
        self.frequencies = sfreq + np.arange(self.nchan * NUPCHAN) * fine_fstep_hz
        
        # find the right index to insert data
        self.start_freq_index = int((sfreq - SFREQ) * NUPCHAN / DEF_CHAN)
        original_time_index = (self.tstart_unix - STIME_UNIX) / TIME_STEP
        if not np.isclose(original_time_index, round(original_time_index)):
            print("Warning: Original time index is not close to an integer.")
        # Round the time index
        self.start_time_index = round(original_time_index)

        # Update the observation attributes for time integration and sample rate if necessary
        obs_group = self.h5file['Observation1']
        obs_group.attrs['tInt'] = self.tstep_s
        self.nframe_read = 0

    def on_data(self, ispan):
        """
        Processes each data span as it arrives, updating the HDF5 file with new intensity data. The method calculates 
        the appropriate index for time insertion based on the accumulated number of frames and updates the intensity 
        dataset within the HDF5 file accordingly.

        Parameters:
            ispan: The data span object, containing the current chunk of data.
        """
        data_cpu = bf.asarray(ispan.data).copy('system').view(np.ndarray)
        data_I = np.abs(data_cpu[:, 0, :, 0, 0, :]).reshape(ispan.nframe, self.nchan * NUPCHAN)  # Assuming nbeam=1
        
        # Update the 'I' dataset
        time_index = self.start_time_index + self.nframe_read
        print('time_index', time_index, 'self.nframe_read', self.nframe_read)
        self.h5file['Observation1/Tuning1/I'][time_index:time_index + ispan.nframe, self.start_freq_index:self.start_freq_index + self.nchan * NUPCHAN] = data_I

        
        self.nframe_read += ispan.nframe

    def shutdown(self):
        if self.h5file:
            self.h5file.close()
            self.h5file = None



class HDF5SaveBlock(bf.pipeline.SinkBlock):
    """
    A SinkBlock for dynamically saving data streams into separate HDF5 files for each beam in 
    real-time. Unlike static pre-allocation, this class creates datasets for intensity ('I'), 
    frequency ('freq'), and time ('time') that grow as data arrives, suitable for varying data sizes 
    and frequency ranges derived from headers.

    This approach allows for more flexible data handling, especially useful in scenarios where
    the data dimensions are not known beforehand and for testing.

    Attributes:
        filename_base (str): Base filename to which beam-specific suffixes will be appended.
        h5files (list): List of open HDF5 file objects, one per beam.
        datasets_I (list): List of 'I' datasets, one per beam.
        datasets_freq (list): List of 'freq' datasets, one per beam.
        datasets_time (list): List of 'time' datasets, one per beam.
        n_iter (int): Counter for iterations or data chunks processed.
        time_dtype (dtype): Data type for the 'time' dataset, including integer and fractional parts.
        nframe_read (int): Number of frames read.
    """
    def __init__(self, iring, filename_base, *args, **kwargs):
        super(HDF5SaveBlock, self).__init__(iring, *args, **kwargs)
        self.filename_base = filename_base
        self.h5files = []
        self.datasets_I = []
        self.datasets_freq = []
        self.datasets_time = []
        self.n_iter = 0
        self.time_dtype = np.dtype([('int', np.int64), ('frac', np.float64)])
        self.nframe_read = 0


    def on_sequence(self, iseq):
        """
        Initializes file and dataset structures based on the first sequence's header information. 
        This includes setting up frequency values, creating HDF5 files for each beam, and preparing 
        datasets.
        
        Parameters:
            iseq: The input sequence from the bifrost pipeline, containing header information.
        """

        self.tstart_unix, self.tstep_s = iseq.header['_tensor']['scales'][0]
        sfreq, fstep_hz = iseq.header['_tensor']['scales'][2]
        fine_fstep_hz = iseq.header['_tensor']['scales'][5][1]
        self.nchan = iseq.header['nchan']
        self.frequencies = sfreq + np.arange(self.nchan*NUPCHAN) * fine_fstep_hz
        self.nbeam = iseq.header['_tensor']['shape'][1]


        for beam in range(self.nbeam):
            filename = f"{self.filename_base}beamdata{beam + 1}.hdf5"
            h5file = h5py.File(filename, 'w')
            self.h5files.append(h5file)
            
            # Creating top-level group 'Observation1'
            obs_group = h5file.create_group('Observation1')
            

            attrs = {
                'ARX_Filter': -1.0,
                'ARX_Gain1': -1.0,
                'ARX_Gain2': -1.0,
                'ARX_GainS': -1.0,
                'Beam': 1,
                'DRX_Gain': -1.0,
                'Dec': -99.0,
                'Dec_Units': 'degrees',
                'Epoch': 2000.0,
                'LFFT': self.nchan*NUPCHAN,
                'RA': -99.0,
                'RA_Units': 'hours',
                'RBW': 23925.78125/NUPCHAN,
                'RBW_Units': 'Hz',
                'TargetName': '',
                'TrackingMode': 'Unknown',
                'nChan': self.nchan*NUPCHAN,
                'sampleRate': 196000000.0,
                'sampleRate_Units': 'Hz',
                'tInt': self.tstep_s, 
                'tInt_Units': 's'
            }


            for attr, value in attrs.items():
                obs_group.attrs[attr] = value



            # Create 'Tuning1' subgroup inside 'Observation1'
            tuning_group = obs_group.create_group('Tuning1')

            # Create datasets 'I' and 'freq' inside 'Tuning1' subgroup
            dataset_I = tuning_group.create_dataset('I', shape=(0, self.nchan * NUPCHAN),
                                                    maxshape=(None, self.nchan * NUPCHAN), dtype=np.float32)
            self.datasets_I.append(dataset_I)

            # Define the frequency dataset properly
            dataset_freq = tuning_group.create_dataset('freq', shape=(self.nchan * NUPCHAN,), dtype=np.float32)
            if self.frequencies is not None:
                dataset_freq[...] = self.frequencies

            # Define the dataset for time based on dimensions you'll extract from header
            dataset_time = obs_group.create_dataset('time', shape=(0,), maxshape=(None,), dtype = self.time_dtype)
            dataset_time.attrs['format'] = 'unix'
            dataset_time.attrs['scale'] = 'utc'
            self.datasets_time.append(dataset_time)

    def on_data(self, ispan):
        """
        Processes and stores incoming data spans into the appropriate 'I' and 'time' datasets, 
        resizing them as needed to fit the new data. This method handles the conversion of 
        data to the correct format and updates datasets for each beam.

        Parameters:
            ispan: The input span from the bifrost pipeline, containing the data.
        """
        in_nframe = ispan.nframe
        print(in_nframe)
        data_cpu = bf.asarray(ispan.data)
        data_cpu_numpy = data_cpu.copy('system').view(np.ndarray)
        
        for beam in range(self.nbeam):
            data_beam = np.abs(data_cpu_numpy[:, beam, :, 0, 0,:]).reshape(ispan.nframe, self.nchan * NUPCHAN)
            
            dataset_I = self.datasets_I[beam]
            dataset_I.resize(dataset_I.shape[0] + data_beam.shape[0], axis=0)
            dataset_I[-data_beam.shape[0]:] = data_beam

            full_timestamps = self.tstart_unix + self.tstep_s * self.nframe_read + np.arange(data_beam.shape[0]) * self.tstep_s
            # Split into integer and fractional parts
            unix_times = np.floor(full_timestamps).astype(np.int64)
            fractions = (full_timestamps - unix_times).astype(np.float64)

            # Create structured array for the 'time' dataset
            time_data = np.array(list(zip(unix_times, fractions)), dtype=self.time_dtype)

            dataset_time = self.datasets_time[beam]
            dataset_time.resize(dataset_time.shape[0] + len(time_data), axis=0)
            dataset_time[-len(time_data):] = time_data
    
        self.nframe_read += in_nframe

    def shutdown(self):
        for h5file in self.h5files:
            if h5file:
                h5file.close()




class AccumHDF5SaveBlock(bf.pipeline.SinkBlock):
    """
    A SinkBlock for accumulating and preprocessing data before saving to HDF5 files. 
    Unlike direct save approaches, this class buffers data, averages it over a specified number 
    of `on_data` calls, and then writes the averaged data to the file. This approach is particularly 
    useful for reducing file size and smoothing out data when high temporal resolution is not required.

    Attributes:
        filename_base (str): Base path for HDF5 files, with beam-specific suffixes added for each file.
        h5files (list): List of HDF5 file objects, one for each beam.
        datasets_I (list): 'I' datasets for storing intensity data, one per beam.
        datasets_freq (list): 'freq' datasets for storing frequency data, one per beam.
        datasets_time (list): 'time' datasets for storing time data, one per beam.
        n_iter (int): Iteration counter for processed data chunks.
        time_dtype (dtype): Structured data type for 'time' dataset, comprising integer and fractional parts.
        nframe_read (int): Counter for the total number of frames read.
        I_buffer (list): Temporary storage for intensity data before averaging.
        time_buffer (list): Temporary storage for time data before averaging.
        buffer_size (int): Current size of the data buffers.
        average_over (int): Number of `on_data` calls over which to average data before writing.
    """
    def __init__(self, iring, filename_base, *args, **kwargs):
        super(AccumHDF5SaveBlock, self).__init__(iring, *args, **kwargs)
        self.filename_base = filename_base
        self.h5files = []
        self.datasets_I = []
        self.datasets_freq = []
        self.datasets_time = []
        self.n_iter = 0
        self.time_dtype = np.dtype([('int', np.int64), ('frac', np.float64)])
        self.nframe_read = 0
        

        self.I_buffer = []
        self.time_buffer = []
        self.buffer_size = 0  # Number of on_data calls accumulated
        self.average_over = 6  # Number of on_data calls to average over


    def on_sequence(self, iseq):
        """
        Prepares HDF5 files and datasets based on sequence header information. Initializes
        buffers for data accumulation and sets dataset attributes based on the incoming
        data's characteristics. Important: some output HDF header parameters are not correct, such as RA and Dec!

        Parameters:
            iseq: Input sequence containing header information.
        """
        print('header in accum block', iseq.header)

        self.tstart_unix, self.tstep_s = iseq.header['_tensor']['scales'][0]
        sfreq, fstep_hz = iseq.header['_tensor']['scales'][2]
        fine_fstep_hz = iseq.header['_tensor']['scales'][5][1]
        self.nchan = iseq.header['nchan']
        self.frequencies = sfreq + np.arange(self.nchan*NUPCHAN) * fine_fstep_hz
        self.nbeam = iseq.header['_tensor']['shape'][1]

        for beam in range(self.nbeam):
            filename = f"{self.filename_base}beamdata{beam + 1}.hdf5"
            h5file = h5py.File(filename, 'w')
            self.h5files.append(h5file)

            # Creating top-level group 'Observation1'
            obs_group = h5file.create_group('Observation1')


            attrs = {
                'ARX_Filter': -1.0,
                'ARX_Gain1': -1.0,
                'ARX_Gain2': -1.0,
                'ARX_GainS': -1.0,
                'Beam': 1,
                'DRX_Gain': -1.0,
                'Dec': -99.0,
                'Dec_Units': 'degrees',
                'Epoch': 2000.0,
                'LFFT': self.nchan*NUPCHAN,
                'RA': -99.0,
                'RA_Units': 'hours',
                'RBW': 23925.78125/NUPCHAN,
                'RBW_Units': 'Hz',
                'TargetName': '',
                'TrackingMode': 'Unknown',
                'nChan': self.nchan*NUPCHAN,
                'sampleRate': 196000000.0,
                'sampleRate_Units': 'Hz',
                'tInt': self.tstep_s*480/NUPCHAN*self.average_over, #For now, should be fixed
                'tInt_Units': 's'
            }
            for attr, value in attrs.items():
                obs_group.attrs[attr] = value



            # Create 'Tuning1' subgroup inside 'Observation1'
            tuning_group = obs_group.create_group('Tuning1')

            # Create datasets 'I' and 'freq' inside 'Tuning1' subgroup
            dataset_I = tuning_group.create_dataset('I', shape=(0, self.nchan * NUPCHAN),
                                                    maxshape=(None, self.nchan * NUPCHAN), dtype=np.float32)
            self.datasets_I.append(dataset_I)

            # Define the frequency dataset properly
            dataset_freq = tuning_group.create_dataset('freq', shape=(self.nchan * NUPCHAN,), dtype=np.float32)
            if self.frequencies is not None:
                dataset_freq[...] = self.frequencies

            # Define the dataset for time based on dimensions you'll extract from header
            dataset_time = obs_group.create_dataset('time', shape=(0,), maxshape=(None,), dtype = self.time_dtype)
            dataset_time.attrs['format'] = 'unix'
            dataset_time.attrs['scale'] = 'utc'
            self.datasets_time.append(dataset_time)

    def on_data(self, ispan):
        """
        Accumulates data from each `on_data` call into buffers. When the buffers reach a specified
        size (`average_over`), averages the buffered data and writes it to the corresponding HDF5 datasets.

        Parameters:
            ispan: Input span from the bifrost pipeline, containing the data chunk to be processed.
        """
        in_nframe = ispan.nframe
        
        data_cpu = bf.asarray(ispan.data)
        data_cpu_numpy = data_cpu.copy('system').view(np.ndarray)

        for beam in range(self.nbeam):
            data_beam = np.abs(data_cpu_numpy[:, beam, :, 0, 0,:]).reshape(ispan.nframe, self.nchan * NUPCHAN)

            # Add to buffer instead of writing directly
            self.I_buffer.append(data_beam)

            full_timestamps = self.tstart_unix + self.tstep_s * self.nframe_read + np.arange(data_beam.shape[0]) * self.tstep_s
            self.time_buffer.append(full_timestamps)

        self.nframe_read += in_nframe
        self.buffer_size += 1


        if self.buffer_size == self.average_over:
            self.average_and_write_data()

    def average_and_write_data(self):
        """
        Averages buffered data and writes the result to HDF5 datasets. This method is triggered
        once the accumulation buffer reaches its target size, as defined by `average_over`.
        Averages are calculated for both intensity and time data, then written to the file.
        """
        # Average the data in the buffers
        concatenated_I = np.concatenate(self.I_buffer, axis=0)
        avg_I = np.mean(concatenated_I, axis=0)


        avg_full_timestamps = np.mean(np.concatenate(self.time_buffer))


        # Split into integer and fractional parts
        unix_times = [np.floor(avg_full_timestamps).astype(np.int64)]
        fractions = [(avg_full_timestamps - unix_times).astype(np.float64)]

        # Recombine into a structured array
        avg_time = np.array(list(zip(unix_times, fractions)), dtype=self.time_dtype)


        for beam in range(self.nbeam):
            # Write averaged data to HDF5 file
            dataset_I = self.datasets_I[beam]
            dataset_I.resize(dataset_I.shape[0] + 1, axis=0)
            dataset_I[-1:] = avg_I

            dataset_time = self.datasets_time[beam]
            dataset_time.resize(dataset_time.shape[0] + 1, axis=0)
            dataset_time[-1:] = avg_time

        # Clear buffers
        self.I_buffer = []
        self.time_buffer = []
        self.buffer_size = 0



    def shutdown(self):
        for h5file in self.h5files:
            if h5file:
                h5file.close()



