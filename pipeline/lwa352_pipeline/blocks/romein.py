from .block_base import Block
from bifrost.proclog import ProcLog
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.device import stream_synchronize, set_device as BFSetGPU
import bifrost.affinity as cpu_affinity
import numpy as np
import ujson as json
import time

from bifrost.romein import Romein

class RomeinNoFFT(Block):
    """ Transform block to grid and image pipeline and collate data
    TODO: How to check?
    Args:
        iring: Ring to read from
        conv: convolutional grid size
        grid: total grid size
    """
    def __init__(self, log, iring, oring, conv, grid, core, nant, gpu=-1):
        super().__init__(log, iring, oring, True, core, etcd_client=None)
        self.iring = iring
        self.oring = oring
        self.grid_size = grid
        self.conv_grid = conv
        self.nant = nant
        self.npol = 2
        self.igulp_size = (nant*(nant-1)) // 2 * self.npol * self.npol * 2 * 4 # one channel of vis data
        self.ogulp_size = self.npol * self.grid_size * self.grid_size  * 8 #complex64

        self.gpu = gpu

    def sendToThreadListener(self, msg):
        self.log.info(msg)

    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        chan_num = 1

        # w-kernel convolutions: just 1s
        # convolution kernel shape: (channels, polarisations, baselines, conv_grid, conv_grid)
        # kernel and out_data need to be same type -> this may need to be complex
        # -> 1 at one space in the grid and zeros elsewhere (0, 0) or (conv_grid/2, conv_grid/2)

        ## TODO: have the npol, nant be passed in as well to avoid magic numbers
        illum = np.zeros(
            shape=(chan_num, 1, (self.nant*(self.nant-1)) // 2, self.npol, self.conv_grid, self.conv_grid),
            dtype=np.complex64
        )
        illum[:, :, :, :, int(self.conv_grid/2), int(self.conv_grid/2)] = 1
        gpu_illum = BFArray(illum, space="cuda")
        illum_size = gpu_illum.dtype.itemsize
        for x in gpu_illum.shape:
            illum_size *= x
        self.log.info("Illum_size: %d bytes, %d MB" % (illum_size, illum_size/1024/1024))

        # Axes: 1 x CHAN [1] x UVW [3] x BASELINES x pol [4]
        uvw = np.random.rand(
            3, 1, chan_num, (self.nant*(self.nant-1)) // 2, self.npol
        )
        #gpu_uvw = BFArray(uvw.transpose(2,0,1,3,4), space="cuda")
        gpu_uvw = BFArray(uvw.astype(np.int32), space="cuda")
        uvw_size = gpu_uvw.dtype.itemsize
        for x in gpu_uvw.shape:
            uvw_size *= x
        self.log.info("uvw size: %d bytes, %d MB" % (uvw_size, uvw_size/1024/1024))

        # out data in shape of (channels, polarisations, grid_size, grid_size)
        self.log.info("Attempting to allocate %d bytes of GPU memory for gridder output, %d MB" % (self.ogulp_size * chan_num, self.ogulp_size * chan_num/1024/1024))
        out_data = BFArray(np.zeros(
            shape=(1, chan_num, self.npol, self.grid_size, self.grid_size),
            dtype=np.complex64),
            space="cuda",
        )

        romein_kernel = Romein()
        print(gpu_uvw.shape)
        print(gpu_illum.shape)
        print(out_data.shape)
        romein_kernel.init(
            gpu_uvw,
            gpu_illum,
            self.grid_size,
            polmajor=False,
        )

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                nchan = ihdr['nchan']
                ohdr = ihdr.copy()
                # Mash header in here if you want
                ohdr_str = json.dumps(ohdr)
                prev_time = time.time()
                chan_id = 0
                num_int = 0
                self.log.info("Starting output sequence with nringlet %d" % iseq.nringlet)
                process_time = 0
                #with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str, nringlet=iseq.nringlet) as oseq:
                for ispan in iseq.read(self.igulp_size * chan_num):
                    idata = ispan.data_view("ci32").reshape(1, chan_num, (self.nant * (self.nant-1))//2, self.npol**2)[:, :, :, :2]#.transpose(0,1,3,2)
                    #self.log.info("Input buffer shape: %s" % str(idata.shape))
                    #self.log.info("Output buffer shape: %s" % str(out_data.shape))
                    if ispan.size < self.igulp_size:
                        continue # Ignore final gulp
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    # idata: 1 x chan (1) x baselines(4)  x pols (4)
                    # odata: 1 x chan (1) x pols (4) x gridx x gridy
                    # 10 GB/s
                    perf = time.perf_counter()
                    romein_kernel.execute(idata, out_data)
                    stream_synchronize()
                    perf = time.perf_counter() - perf
                    #self.log.info("channel %d: Timing for %d channel gridding: %f" % (chan_id, chan_num, perf))
                    chan_id += chan_num
                    process_time += perf
                    if chan_id == nchan:
                        chan_id = 0
                        num_int += 1
                        self.log.info("%d: Gridding completed for %d channels" % (num_int, nchan))
                        total_bytes = 8 * nchan * self.npol * (self.nant*(self.nant-1)//2)
                        self.log.info("%d bytes processed in %fs (average %fs per gridding stream) = throughput GBps: %f" % (total_bytes, process_time, (process_time/nchan*chan_num), (total_bytes/1024/1024/1024/process_time)))
                        process_time = 0
                        #with oseq.reserve(self.ogulp_size) as ospan:
                        #    copy_array(ospan.data, out_data)
                        #if (self.oring.space == 'cuda') or (self.iring.space=='cuda'):
                        #    stream_synchronize()

