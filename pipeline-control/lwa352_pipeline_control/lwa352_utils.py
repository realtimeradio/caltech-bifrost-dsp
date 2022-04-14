# System parameters
FS_HZ = 196000000 # ADC sample rate in Hz
NCHAN = 4096      # Number of channels generated by the F-engines

def time_to_spectra(t):
    """
    Given a UNIX time, return a spectra count
    since the UNIX epoch.
    """
    sample_number = int(t * FS_HZ)
    spectra_number = sample_number // (2*NCHAN)
    return spectra_number

def spectra_to_time(s):
    """
    Given a spectra ID (spectra count since UNIX
    epoch) return a UNIX time
    """
    sample_number = s * 2 * NCHAN
    sample_time = sample_number / FS_HZ
    return sample_time
