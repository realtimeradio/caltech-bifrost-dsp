import bifrost as bf
from bifrost.libbifrost import _bf                                                                                                                                               
import time
import numpy as np

NSTATION=352; NFREQUENCY=192; NTIME=512;
DOSIM=False
#NSTATION=32; NFREQUENCY=32; NTIME=32;
MATLEN = 47849472

## Computes the triangular index of an (i,j) pair as shown here...
## NB: Output is valid only if i >= j.
##
##      i=0  1  2  3  4..
##     +---------------
## j=0 | 00 01 03 06 10
##   1 |    02 04 07 11
##   2 |       05 08 12
##   3 |          09 13
##   4 |             14
##   :
def tri_index(i, j):
    return (i * (i+1))//2 + j;

## Returns index into the GPU's register tile ordered output buffer for the
## real component of the cross product of inputs in0 and in1.  Note that in0
## and in1 are input indexes (i.e. 0 based) and often represent antenna and
## polarization by passing (2*ant_idx+pol_idx) as the input number (NB: ant_idx
## and pol_idx are also 0 based).  Return value is valid if in1 >= in0.  The
## corresponding imaginary component is located xgpu_info.matLength words after
## the real component.
def regtile_index(in0, in1):
    a0 = in0 >> 1;
    a1 = in1 >> 1;
    p0 = in0 & 1;
    p1 = in1 & 1;
    num_words_per_cell = 4;
  
    # Index within a quadrant
    quadrant_index = tri_index(a1//2, a0//2);
    # Quadrant for this input pair
    quadrant = 2*(a0&1) + (a1&1);
    # Size of quadrant
    quadrant_size = (NSTATION//2 + 1) * NSTATION//4;
    # Index of cell (in units of cells)
    cell_index = quadrant*quadrant_size + quadrant_index;
    #printf("%s: in0=%d, in1=%d, a0=%d, a1=%d, cell_index=%d\n", __FUNCTION__, in0, in1, a0, a1, cell_index);
    # Pol offset
    pol_offset = 2*p1 + p0;
    # Word index (in units of words (i.e. floats) of real component
    index = (cell_index * num_words_per_cell) + pol_offset;
    return index;


SPACE='cuda'

invec = np.ones([NTIME, NFREQUENCY, NSTATION, 2])
if DOSIM:
    print("Polulating test vectors")
    for t in range(NTIME):
        for s in range(NSTATION):
            invec[:,:,s,:] = (s+1)%7
else:
    print("Not populating test vectors")

print('allocating input')
ibuf = bf.ndarray(invec, dtype='ci4', space=SPACE)
print('allocating output')
obuf = bf.ndarray(np.zeros([MATLEN], dtype=np.int32), dtype='ci32', space=SPACE)
#obuf = bf.ndarray(np.zeros([NFREQUENCY, MATLEN//NFREQUENCY], dtype=np.int32), dtype='ci32', space=SPACE)
print(obuf[0:10])

if SPACE == 'cuda':
    print('running kernel as_GPUarray')
    _bf.xgpuInitialize(ibuf.as_BFarray(), obuf.as_BFarray(), 0)
    for i in range(4):
        _bf.xgpuKernel(ibuf.as_BFarray(), obuf.as_BFarray(), 0)
    _bf.xgpuKernel(ibuf.as_BFarray(), obuf.as_BFarray(), 1)
    print(obuf[0:10])
else:
    print('running kernel as_BFarray')
    _bf.xgpuInitialize(ibuf.as_BFarray(), obuf.as_BFarray(), 0)
    for i in range(4):
        _bf.xgpuCorrelate(ibuf.as_BFarray(), obuf.as_BFarray(), 0)
    _bf.xgpuCorrelate(ibuf.as_BFarray(), obuf.as_BFarray(), 1)
    print(obuf[0:10])

obuf_cpu = obuf.copy(space='system')
print('copied')
#view as real/imag x chan x station
o = obuf_cpu.view(dtype=np.int32).reshape(2, NFREQUENCY, MATLEN//NFREQUENCY)
oc = o[0,0,:] + 1j*o[1,0,:]

acc_len = 5 * NTIME
for s0 in range(5):
    for s1 in range(s0, 5):
        ar = ibuf[0,0,s0,0].real[0]
        ai = ibuf[0,0,s0,0].imag[0]
        a = ar + 1j*ai
        br = ibuf[0,0,s1,0].real[0]
        bi = ibuf[0,0,s1,0].imag[0]
        b = br + 1j*bi
        v = a * np.conj(b)
        v *= acc_len
        print(s0, s1, v, oc[regtile_index(2*s0,2*s1)])

#from matplotlib import pyplot as plt
#plt.plot(o[0,0,:])
#plt.plot(o[1,0,:])
#plt.show()
