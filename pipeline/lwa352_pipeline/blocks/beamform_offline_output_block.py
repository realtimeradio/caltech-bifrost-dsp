from bifrost.pipeline import SinkBlock
import h5py
import numpy as np
import bifrost as bf

class HDF5SaveBlock(bf.pipeline.SinkBlock):
    def __init__(self, iring, filename, *args, **kwargs):
        super(HDF5SaveBlock, self).__init__(iring, *args, **kwargs)
        self.filename = filename
        self.h5file = None
        self.dataset = None
        np.set_printoptions(threshold=np.inf)
        self.n_iter = 0

    def on_sequence(self, iseq):
        #  If the file already exists, it'll be overwritten.
        self.h5file = h5py.File(self.filename, 'w')
        # Create an expandable dataset in the HDF5 file.
        # The initial size is set to (0, 96*32*1), more beams TBD
        self.dataset = self.h5file.create_dataset('data', shape=(0, 96*32*1),
                                                  maxshape=(None, 96*32*1), dtype=np.float32)
        # Add  header to an HDF5
        #self.dataset.attrs['header'] = str(iseq.header)

    def on_data(self, ispan):
        data_cpu = bf.asarray(ispan.data)


        data_cpu_numpy = data_cpu.copy('system').view(np.ndarray)
        data_real = np.abs(data_cpu_numpy).reshape(ispan.nframe, 96*32*1)

        print('data real shape', np.shape(data_real))
        print('n_iter', self.n_iter)
        #print('actua; data', data_real)


        self.n_iter += 1
        self.dataset.resize(self.dataset.shape[0] + data_real.shape[0], axis=0)
        self.dataset[-data_real.shape[0]:] = data_real

    def shutdown(self):
        if self.h5file:
            self.h5file.close()


