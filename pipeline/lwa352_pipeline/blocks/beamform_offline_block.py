from copy import deepcopy
import numpy as np
from bifrost.pipeline import TransformBlock
from bifrost.linalg import LinAlg
from astropy.constants import c as speedOfLight
speedOfLight = speedOfLight.to('m/ns').value

from lwa_antpos.station import ovro
from bifrost import ndarray as BFArray

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import Angle


linalg = LinAlg()

NPOL = 2
NUPCHAN = 32

class BfOfflineBlock(TransformBlock):
    def __init__(self, iring, nbeam, nbeams_per_batch, ntimestep, ra_array, dec_array, cal_data, station=ovro, frame_size=1, *args, **kwargs):
        super(BfOfflineBlock, self).__init__(iring, *args, **kwargs)
        self.nbeam = nbeam # Number of beams to form
        self.nbeams_per_batch = nbeams_per_batch
        if nbeam % nbeams_per_batch != 0:
            raise ValueError(f"Total number of beams ({nbeam}) must be divisible by number of beams per batch ({nbeams_per_batch})")

        self.ntimestep = ntimestep # Number of time samples between coefficient updates
        self.station = station # station
        self.frame_size = frame_size  # Frame size
        self.ra_array = ra_array  # Right Ascension array
        self.dec_array = dec_array  # Declination array

        if cal_data:
            self.cal_data = cal_data  
        else:
            print("Warning: Gain calibration data not available!")
        
        assert nbeam == nbeams_per_batch, "Number of batches and beams should be the same for correct usage for now"
        assert len(self.ra_array) == self.nbeam, "Mismatch in number of ra angles and beams"
        assert len(self.dec_array) == self.nbeam, "Mismatch in number of dec angles and beams"

        # Initially set uniform antenna weighting for a natural beam shape
        self.set_beam_weighting(lambda x: 1.0)
        



    def set_beam_target(self, ra, dec, observation_time, verbose=True):
        """
        Given a target's RA and DEC (degrees), and the time of observation, compute the
        topocentric position of the target and point the beam at it.
        """

        # Load in where we are
        obs = EarthLocation.from_geocentric(*self.station.ecef, unit=u.m)

        # Convert RA and DEC to SkyCoord object
        ra = Angle(ra, unit=u.deg)
        dec = Angle(dec, unit=u.deg)
        sc = SkyCoord(ra, dec, frame='icrs')

        # Figure out where it is at the given observation time
        compute_time = Time(observation_time, format='unix', scale='utc')
        aa = sc.transform_to(AltAz(obstime=compute_time, location=obs))
        az = aa.az.to(u.rad).value  # Convert to radians
        alt = aa.alt.to(u.rad).value  # Convert to radians
        if verbose:
            print(f"At observation time, target is at azimuth {aa.az}, altitude {aa.alt}")

        # Return azimuth and altitude in radians
        return az, alt

    def set_beam_weighting(self, fnc=lambda x: 1.0):
        """
        Set the beamformer antenna weighting using the provided function.  The
        function should accept a single floating point input of an antenna's
        distance from the array center (in meters) and return a weight between
        0 and 1, inclusive.
        """
        fnc2 = lambda x: np.clip(fnc(np.sqrt(x[0] ** 2 + x[1] ** 2)), 0, 1)
        self._weighting = np.array([fnc2(ant.enz) for ant in self.station.antennas])
        self._weighting = np.repeat(self._weighting, 2)

    def compute_weights(self, observation_time, batch_start, batch_end):

        batch_ra_array = self.ra_array[batch_start:batch_end]
        batch_dec_array = self.dec_array[batch_start:batch_end]

        weights_array = []
        for beam in range(self.nbeams_per_batch):
            ra = batch_ra_array[beam]
            dec = batch_dec_array[beam]

            # Set beam target and get az and alt values
            az, alt = self.set_beam_target(ra, dec, observation_time)

            assert (az >= 0 and az < 2 * np.pi)
            assert (alt >= 0 and alt <= np.pi / 2)

            zenith = np.array([0, 0, 1])
            zenith_delay = [np.dot(zenith, antenna.enz) / speedOfLight for antenna in self.station.antennas]

            direction = np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
            direction_delay = [np.dot(direction, antenna.enz) / speedOfLight for antenna in self.station.antennas]

            delays = np.array(direction_delay) - np.array(zenith_delay)
            delays = np.repeat(delays, NPOL)
            delays = delays.max() - delays



            weighted_delays = self._weighting * delays

            weights = np.exp(2j * np.pi * self.frequencies[:, None] * weighted_delays * 1e-9)
            weights_array.append(weights)


        return np.array(weights_array)

    def on_sequence(self, iseq):
        """
        At the start of a sequence, figure out how many stands / pols / chans
        we are dealing with, and construct an array for coefficients.
        """
 
        #print("header after upchan:", iseq.header)
        # Extract parameters to be used in the beamforming process from the header
        self.tstart_unix, self.tstep_s = iseq.header['_tensor']['scales'][0]
        self.gulp_nframe = iseq.header['_tensor']['gulp_nframe']
        sfreq, fstep_hz = iseq.header['_tensor']['scales'][1]
        fine_fstep_hz = iseq.header['_tensor']['scales'][4][1]
        self.frequencies = sfreq + np.arange(iseq.header['system_nchan']) * fine_fstep_hz
        self.nchan = iseq.header['nchan']
        self.nstand = iseq.header['nstand']
        self.npol = NPOL
        # Extract the relevant index for calibration data to apply before beamforming 
        if hasattr(self, 'cal_data'):
            try:
                self.index = self.cal_data['frequencies'].index(self.frequencies[0])
            except ValueError:
                print(f"Warning: No calibration data available for frequency {self.frequencies[0]}!")

        ohdr = deepcopy(iseq.header)

        # Create empty coefficient array
        self.coeffs = BFArray(
            np.zeros([self.nbeams_per_batch, self.nchan * NUPCHAN, self.nstand * self.npol], dtype=complex),
            space='cuda_host')
        
        self.calibration_data = BFArray(self.cal_data['data'][self.index].transpose(1, 0, 2, 3).reshape(1, self.nchan, self.nstand, self.npol, NUPCHAN), space='cuda_host')


        # Manipulate header. Dimensions will be different (beams, not stands)
        ohdr['_tensor'] = {
            'dtype':  'cf32',  
            'shape': [-1, self.nchan, NUPCHAN, self.nbeam],
            'labels': ['time', 'freq', 'fine_freq', 'beam',],
            'scales': [(self.tstart_unix, self.tstep_s * self.frame_size),
                       (sfreq, fstep_hz),
                       (0, fine_fstep_hz),
                       None,
                       None],
            'units': ['s', 'HZ', '1/s', None, None],
            'gulp_nframe': self.gulp_nframe,
        }

        self.nframe_read = 0
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe = ispan.nframe # Number of frames to read
        out_nframe = in_nframe # Probably don't accumulate in this block
        
        idata = ispan.data
        #idata shape (15, 96, 352, 2, 32)
        odata = ospan.data
        
        # Convert to BFArray 
        #calibration_data_bf = BFArray(calibration_data, space='cuda_host')
        idata = BFArray(idata,space='cuda_host')
        # Pre-compute multiplied data if multiplication remains constant for each frame
        idata = idata * self.calibration_data
        # Calculate the observation time for the current gulp
        gulp_start_time = self.tstart_unix + self.tstep_s * self.nframe_read
        print(self.nframe_read)
        for batch_start in range(0, self.nbeam, self.nbeams_per_batch):
            batch_end = min(batch_start + self.nbeams_per_batch, self.nbeam)
            
            for i in range(in_nframe):
                if self.nframe_read % self.ntimestep == 0:
                    observation_time = gulp_start_time + self.tstep_s * i
                    print("obs time", observation_time)
                    self.coeffs = self.compute_weights(observation_time, batch_start, batch_end)
                    # Manipulate dimensions of coeffs and idata to produce beamforming results                    
                    self.coeffs = self.coeffs.reshape(self.nbeams_per_batch, self.nchan, NUPCHAN, self.nstand * self.npol)
                    self.coeffs = self.coeffs.transpose(1, 2, 3, 0)
                    self.coeffs = self.coeffs.astype(np.complex64)

                idata_reshaped = idata[i].reshape(self.nchan, self.nstand * self.npol, NUPCHAN)
                idata_transposed = idata_reshaped.transpose(0, 2, 1)
                idata_expanded = np.expand_dims(idata_transposed, axis=-1)

                    #self.coeffs = self.coeffs.astype(np.complex64)
                                        
                idata_expanded_reshaped = idata_expanded.repeat(self.nbeams_per_batch, axis=-1) #if not done, broadcasting causes stride errors

                odata[i,:,:,batch_start:batch_end] = np.sum(self.coeffs * idata_expanded_reshaped, axis=2)

                self.nframe_read += 1
                
        return out_nframe
