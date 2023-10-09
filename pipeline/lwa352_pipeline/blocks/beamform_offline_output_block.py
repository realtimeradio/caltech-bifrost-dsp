from bifrost.pipeline import SinkBlock
import h5py
import numpy as np
import bifrost as bf

class HDF5SaveBlock(bf.pipeline.SinkBlock):
    def __init__(self, iring, filename_base, *args, **kwargs):
        super(HDF5SaveBlock, self).__init__(iring, *args, **kwargs)
        self.filename_base = filename_base
        self.h5files = []
        self.datasets_I = []
        self.datasets_freq = []
        self.datasets_time = []
        self.n_iter = 0

    def on_sequence(self, iseq):
        self.nbeam = iseq.header['_tensor']['shape'][3]
        
        for beam in range(self.nbeam):
            filename = f"{self.filename_base}beamdata{beam + 1}.hdf5"
            h5file = h5py.File(filename, 'w')
            self.h5files.append(h5file)
            
            # Creating top-level group 'Observation1'
            obs_group = h5file.create_group('Observation1')

            # Create 'Tuning1' subgroup inside 'Observation1'
            tuning_group = obs_group.create_group('Tuning1')

            # Create datasets 'I' and 'freq' inside 'Tuning1' subgroup
            dataset_I = tuning_group.create_dataset('I', shape=(0, 96*32),
                                                    maxshape=(None, 96*32), dtype=np.float32)
            self.datasets_I.append(dataset_I)

            # TODO: Define the frequency dataset properly
            dataset_freq = tuning_group.create_dataset('freq', shape=(96*32,), dtype=np.float32)
            self.datasets_freq.append(dataset_freq)

            # TODO: Define the dataset for time based on dimensions you'll extract from header
            time_shape = (0,)  # dummy shape, update as required
            dataset_time = obs_group.create_dataset('time', shape=time_shape, maxshape=(None,))
            self.datasets_time.append(dataset_time)

    def on_data(self, ispan):
        data_cpu = bf.asarray(ispan.data)
        data_cpu_numpy = data_cpu.copy('system').view(np.ndarray)
        
        for beam in range(self.nbeam):
            data_beam = np.real(data_cpu_numpy[:, :, :, beam]).reshape(ispan.nframe, 96*32)
            
            dataset_I = self.datasets_I[beam]
            dataset_I.resize(dataset_I.shape[0] + data_beam.shape[0], axis=0)
            dataset_I[-data_beam.shape[0]:] = data_beam

            # TODO: Handle the 'freq' and 'time' datasets as necessary for each beam

    def shutdown(self):
        for h5file in self.h5files:
            if h5file:
                h5file.close()

'''class HDF5SaveBlock(bf.pipeline.SinkBlock):
    def __init__(self, iring, filename, *args, **kwargs):
        super(HDF5SaveBlock, self).__init__(iring, *args, **kwargs)
        self.filename = filename
        self.h5file = None
        self.dataset_I = None
        self.n_iter = 0

    def on_sequence(self, iseq):
        self.nbeam = iseq.header['_tensor']['shape'][3]
        # Open a single HDF5 file
        self.h5file = h5py.File(self.filename, 'w')
        obs_group = self.h5file.create_group('Observation1')
        tuning_group = obs_group.create_group('Tuning1')

        # Create a single 'I' dataset with an extra dimension for beams
        self.dataset_I = tuning_group.create_dataset('I', shape=(0, self.nbeam, 96*32),
                                                     maxshape=(None, self.nbeam, 96*32), dtype=np.float32)
        # TODO: Define the frequency dataset properly
        self.dataset_freq = tuning_group.create_dataset('freq', shape=(96*32,), dtype=np.float32)

        # Create 'time' dataset
        time_shape = (0,)  # dummy shape, update as required
        self.dataset_time = obs_group.create_dataset('time', shape=time_shape, maxshape=(None,))

    def on_data(self, ispan):
        data_cpu = bf.asarray(ispan.data)
        data_cpu_numpy = data_cpu.copy('system').view(np.ndarray)
        data_real = np.real(data_cpu_numpy).reshape(ispan.nframe, self.nbeam, 96*32)

        self.dataset_I.resize(self.dataset_I.shape[0] + data_real.shape[0], axis=0)
        self.dataset_I[-data_real.shape[0]:] = data_real

        # TODO: Handle the 'freq' and 'time' datasets as necessary

    def shutdown(self):
        if self.h5file:
            self.h5file.close()
'''
