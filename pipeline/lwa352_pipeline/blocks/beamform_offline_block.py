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
class BfOfflineBlock(TransformBlock):
    def __init__(self, iring, nbeam, ntimestep, ra_array, dec_array, station=ovro,frame_size=1, *args, **kwargs):
        super(BfOfflineBlock, self).__init__(iring, *args, **kwargs)
        self.nbeam = nbeam # Number of beams to form
        self.ntimestep = ntimestep # Number of time samples between coefficient updates
        self.station = station # station
        self.frame_size = frame_size  # Frame size
        self.ra_array = ra_array  # Right Ascension array
        self.dec_array = dec_array  # Declination array

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
        assert len(self.ra_array) == self.nbeam, "Mismatch in number of ra angles and beams"
        assert len(self.dec_array) == self.nbeam, "Mismatch in number of dec angles and beams"

        weights_array = []
        for beam in range(self.nbeam):
            ra = self.ra_array[beam]
            dec = self.dec_array[beam]

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
        # Get the observation time from the tensor header
        self.tstart_unix, self.tstep_s = iseq.header['_tensor']['scales'][0]
        self.gulp_nframe = iseq.header['_tensor']['gulp_nframe']
        sfreq, fstep_hz = iseq.header['_tensor']['scales'][2]
        self.frequencies = sfreq + np.arange(iseq.header['nchan']) * fstep_hz


        ohdr = deepcopy(iseq.header)

        # Create empty coefficient array
        self.coeffs = BFArray(
            np.zeros([self.nbeam, iseq.header['nchan'], iseq.header['nstand'], iseq.header['npol']], dtype=complex),
            space='cuda_host')
        # Manipulate header. Dimensions will be different (beams, not stands)
        ohdr['_tensor'] = {
            'dtype': 'ci' + str(ohdr['nbit']),
            'shape': [-1, self.frame_size, ohdr['nchan'], self.nbeam, ohdr['npol']],
            'labels': ['time', 'fine_time', 'freq', 'beam', 'pol'],
            'scales': [(self.tstart_unix, self.tstep_s * self.frame_size),
                       (0, self.tstep_s),
                       (ohdr['sfreq'], fstep_hz),
                       None,
                       None],
            'units': ['s', 's', 'Hz', None, None],
            'gulp_nframe': self.gulp_nframe,
        }

        self.nframe_read = 0
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe = ispan.nframe # Number of frames to read
        out_nframe = in_nframe # Probably don't accumulate in this block

        idata = ispan.data
        odata = ospan.data

        # Calculate the observation time for the current gulp
        gulp_start_time = self.tstart_unix + self.tstep_s * ispan.sequence

        # Update the coefficients every ntimestep
        for i in range(in_nframe):
            if self.nframe_read % self.ntimestep == 0:
                # Compute the observation time for the current frame
                observation_time = gulp_start_time + self.tstep_s * i / self.gulp_nframe
                self.coeffs = self.compute_weights(observation_time)

                # Reshape to (nbeam, nchan, nstand, npol)
                self.coeffs = self.coeffs.reshape(self.nbeam, self.nchan, self.nstand, self.npol)
                # Swap nchan and nbeam
                self.coeffs = self.coeffs.transpose(1, 0, 2, 3)

            for j in range(self.frame_size):
                # Perform element-wise multiplication and then sum over the nstand axis
                odata[i, j] = np.sum(self.coeffs * idata[i, j][:, np.newaxis, :, :], axis=2)


            ##linalg(1, idata[i], self.coeffs, 0, odata[i])

            self.nframe_read += 1

        return out_nframe
