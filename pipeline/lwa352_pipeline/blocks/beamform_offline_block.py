from copy import deepcopy
import numpy as np
from bifrost.pipeline import TransformBlock
from bifrost.linalg import LinAlg
from bifrost import ndarray as BFArray

linalg = LinAlg()

class BfOfflineBlock(TransformBlock):
    def __init__(self, iring, nbeam, ntimestep, ant_locs, *args, **kwargs):
        super(BfOfflineBlock, self).__init__(iring, *args, **kwargs)
        self.nbeam = nbeam # Number of beams to form
        self.ntimestep = ntimestep # Number of time samples between coefficient updates
        self.ant_locs = ant_locs # Antenna locations

    def compute_coefficients(self, time):
        # Take time, antenna locations; update coefficients
        # self.coeffs = <....>
        pass

    def on_sequence(self, iseq):
        """
        At the start of a sequence, figure out how many stands / pols / chans
        we are dealing with, and construct an array for coefficients.
        """
        ohdr = deepcopy(iseq.header)
        # Create empty coefficient array
        self.coeffs = BfArray(np.zeros([hdr['nchan'], hdr['nstand'], hdr['npol']], dtype=complex), space='cuda_host')
        # Manipulate header. Dimensions will be different (beams, not stands)
        #hdr['_tensor'] = {
        #    }
        self.nframe_read = 0
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe = ispan.nframe # Number of frames to read
        out_nframe = in_nframe # Probably don't accumulate in this block

        idata = ispan.data
        odata = ospan.data

        linalg(1, idata, self.coeffs, 0, odata) # These arguments aren't right, but there is a matrix multiply here
        self.nframe_read += in_nframe
        some_time_s
        self.compute_coefficients(self.nframe_read * some_time_scale) # FIXME

        return out_nframe
