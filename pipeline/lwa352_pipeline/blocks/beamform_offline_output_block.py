from bifrost.pipeline import SinkBlock
import h5py
import numpy as np
import bifrost as bf


NUPCHAN = 32

class HDF5SaveBlock(bf.pipeline.SinkBlock):
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



