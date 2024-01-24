from copy import deepcopy
import numpy as np
from bifrost.pipeline import TransformBlock
from bifrost.linalg import LinAlg
from astropy.constants import c as speedOfLight
speedOfLight = speedOfLight.to('m/ns').value

from lwa_antpos.station import ovro
from bifrost import ndarray as BFArray
from bifrost import map as BFmap
from bifrost import empty as BFempty

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
from astropy.coordinates import Angle

linalg = LinAlg()

NPOL = 2
NUPCHAN = 32

class BfOfflineWeightsBlock(TransformBlock):
    def __init__(self, iring, nbeam, ntimestep, ra_array, dec_array, cal_data, station=ovro, frame_size=1, *args, **kwargs):
        super(BfOfflineWeightsBlock, self).__init__(iring, *args, **kwargs)
        self.nbeam = nbeam # Number of beams to form

        self.ntimestep = ntimestep # Number of time samples between coefficient updates
        self.station = station # station
        self.frame_size = frame_size  # Frame size
        self.ra_array = ra_array  # Right Ascension array
        self.dec_array = dec_array  # Declination array

        if cal_data:
            self.cal_data = cal_data  
        else:
            print("Warning: Gain calibration data not available!")
        
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

    def compute_weights(self, observation_time):
        """
        Compute weights associated with geometric delays towards a direction with
        predefined RA and Dec using lwa antenna positions.
        """
        weights_array = []
        for beam in range(self.nbeam):
            ra = self.ra_array[beam]
            dec = self.dec_array[beam]
            
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
 
        # Extract parameters to be used in the beamforming process from the header
        self.tstart_unix, self.tstep_s = iseq.header['_tensor']['scales'][0]
        self.gulp_nframe = iseq.header['_tensor']['gulp_nframe']
        sfreq, fstep_hz = iseq.header['_tensor']['scales'][1]
        fine_fstep_hz = iseq.header['_tensor']['scales'][4][1]
        self.nchan = iseq.header['nchan']
        self.frequencies = sfreq + np.arange(self.nchan * NUPCHAN) * fine_fstep_hz
        self.nstand = iseq.header['nstand']
        self.npol = NPOL
        # Extract the relevant index for calibration data to apply before beamforming 
        if hasattr(self, 'cal_data'):
            try:
                self.index = self.cal_data['frequencies'].index(self.frequencies[0])
            except ValueError:
                print(f"Warning: No calibration data available for frequency {self.frequencies[0]}!")

        ohdr = deepcopy(iseq.header)

    

        # Reshape cal_data to introduce an additional axis for nbeam. 
        # This ensures it's broadcast-compatible with idata 
        # initially caldata is (nstand, nchan, nupchan, npol) --> (nchan,nstand,npol,nupchan) --> expanding dimesnions to be compatible with idata
        cal_data_arr = np.array(self.cal_data['data'][self.index].transpose(1, 0, 3, 2).reshape(1, 1, self.nchan, self.nstand, self.npol, NUPCHAN))
        # Keep it on cpu since trasferring to cuda here changes values
        self.calibration_data = BFArray(cal_data_arr, space='cuda_host')

        # Manipulate header

        ohdr['_tensor'] = {
            'dtype':  'cf32',
            'shape': [-1, self.nbeam, self.nchan, self.nstand, self.npol,  NUPCHAN],
            'labels': ['time',' beam', 'freq', 'stand', 'pol',  'fine_freq'],
            'scales': [(self.tstart_unix, self.tstep_s * self.frame_size),
                       None,
                       (sfreq, fstep_hz),
                       (None,1),
                       (None,1),
                       (0, fine_fstep_hz),
                        ],
            'units': ['s', None, 'Hz', None, None, '1/s',],
            'gulp_nframe': self.gulp_nframe,
        }

        self.nframe_read = 0
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe = ispan.nframe # Number of frames to read
        out_nframe = in_nframe # Probably don't accumulate in this block
        
        idata = ispan.data
        odata = ospan.data

        # Expand idata so that nbeam is formed simultaneously
        idata = idata[:, np.newaxis, ...]
        

        #TODO repeat of idata for nbeam>1
        #idata = idata.repeat(self.nbeam, axis=1)
        
        # Applying weights obtained from caltables
        idata = BFArray(idata,space='cuda')
        #self.calibration_data = BFArray(self.calibration_data, space='cuda')
        #BFmap("idata = idata * cal", {'idata': idata, 'cal': self.calibration_data})
        
        # Calculating the observation time for the data
        gulp_start_time = self.tstart_unix + self.tstep_s * self.nframe_read
        print(self.nframe_read)
        
        # Check if it's time to update the geometric weights based on the predefined number of timesteps 
        if self.nframe_read % self.ntimestep == 0:
            observation_time = gulp_start_time 
            print("obs time", observation_time)
            self.coeffs = self.compute_weights(observation_time)
            # Manipulate dimensions of coeffs and idata (below) to produce beamforming results
            #self.coeffs = self.coeffs.reshape(self.nbeam, self.nchan, NUPCHAN, self.nstand * self.npol)
            self.coeffs = self.coeffs.reshape(self.nbeam, self.nchan, NUPCHAN, self.nstand , self.npol)
            self.coeffs = self.coeffs[np.newaxis,...]
            # Repeat geometric delay coeffs to match idata nframe
            self.coeffs = self.coeffs.repeat(in_nframe, axis=0)
            self.coeffs = self.coeffs.astype(np.complex64)
            self.coeffs = self.coeffs.transpose(0,1,2,4,5,3)   
            


        full_cal_data = self.calibration_data * self.coeffs

        full_cal_data = BFArray(full_cal_data, space='cuda')
        
        BFmap("a = a * b", {'a': idata, 'b': full_cal_data})
        
        odata[...] = idata

        self.nframe_read += in_nframe


        return out_nframe


