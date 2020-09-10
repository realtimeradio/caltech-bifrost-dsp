
```
usage: make_golden_inputs.py [-h] [-t NTIME] [-c NCHAN] [-s NSTAND] [-p NPOL]
                             [--accshort ACCSHORT] [--seed SEED] [--nocorr]
                             [--datapath DATAPATH] [--chanramp]

Script for generating golden input / output files

optional arguments:
  -h, --help            show this help message and exit
  -t NTIME, --ntime NTIME
                        Number of time samples for which data should be
                        generated (default: 480000)
  -c NCHAN, --nchan NCHAN
                        Number of freq channels for which data should be
                        generated (default: 192)
  -s NSTAND, --nstand NSTAND
                        Number of stands for which data should be generated
                        (default: 352)
  -p NPOL, --npol NPOL  Number of polarizations for which data should be
                        generated (default: 2)
  --accshort ACCSHORT   Number of samples to accumulate for fast correlations
                        (default: 2400)
  --seed SEED           Seed for random number generation (default:
                        3735928559)
  --nocorr              Do not generate correlation files (default: False)
  --datapath DATAPATH   Directory in which to put data (default: /data/)
  --chanramp            Make all test vectors a ramp with channel number
                        (default: False)
```

Generate two test files.

The first:
```
in_<ntime>t_<nchan>c_<nstand>s_<npol>p_<seed>.dat
```
is an `ntime` x `nchan` x `nstand` x `npol` x 4+4 bit array of either random numbers with seed `seed` or a
channel ramp where all the 8-bit value of each sample == the channel number. In this case `seed` in the file name
is `"chanramp"`.

Data follows a one-line JSON encoded dictionary header with the fields:

- time : The UNIX time the generation script was run
- ntime : The number of time samples in the file
- nstand : The number of antennas in the file
- npol : The number of polarization in the file
- nchan : The number of frequency channels in the file
- seed : The seed used for random number generation
- shape : A list of array axis sizes for the data set; i.e. [ntime, nchan, nstand, npol]
- dtype : String representation of the data type; i.e. "np.uint8"
- type : "random" if the dataset is random numbers, "chanramp" (or other future TBC descriptor) otherwise.

The second:
```
corr_<ntime>t_<accshort>a_<nchan>c_<nstand>s_<npol>p_<seed>.dat
```
is an `ntime // accshort` x `nchan` x `nstand` x `nstand` x `npol` x `npol` array of data type `numpy.complex`

constructed by performing a correlation of the test samples in the first file and integrating over `accshort` spectra.
The correlation convention is that `output[:, :, stand0, stand1, pol0, pol1]` is `stand0, pol0 x conj(stand1, pol1)`

Data follows a one-line JSON encoded dictionary header with the fields:

- time : The UNIX time the generation script was run
- acc_len: The number of time samples integrated together in this file
- ntime : The number of time samples in the file (after integration)
- nstand : The number of antennas in the file
- npol : The number of polarization in the file
- nchan : The number of frequency channels in the file
- seed : The seed used for random number generation
- shape : A list of array axis sizes for the data set; i.e. [ntime, nchan, nstand, nstand, npol, npol]
- dtype : String representation of the data type; i.e. "np.complex"
- type : "random" if the dataset is random numbers, "chanramp" (or other future TBC descriptor) otherwise.
