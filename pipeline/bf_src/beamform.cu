#include <stdio.h>
#include <time.h>

#define NPOLS (2*64)
#define NCHANS 256
#define NTIMES 16
#define NBEAMS 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

/* Beamforming kernel
 * Launch with block dimensions: n_times x n_beams;
 * with n_chan threads per block
 * Input data is in order: time x chan x pol
 * Weights are in order: chan x beam x pol
 * Outputs are in order chan x time x beam
 */

//__global__
//void beamform(float *weights, char *in, float *out, int npols, int nchans)
//{
//  int ntimes = blockDim.x;
//  int nbeams = blockDim.y;
//  int t = blockIdx.x;
//  int b = blockIdx.y;
//  int c = threadIdx.x;
//  
//  int p;
//  float out_r = 0.0;
//  float out_i = 0.0;
//  char pr, pi;
//  float wr, wi;
//  char *in_block = in + 2*(t*nchans*npols + c*npols);
//  float *weight_block = weights + 2*(c*npols*nbeams + b*npols);
//  float *out_block = out + 2*(c*ntimes*nbeams + t*nbeams + b);
//  for (p=0; p<npols; p++){
//    pr = in_block[2*p];
//    pi = in_block[2*p+1];
//    //wr = weight_block[2*p];
//    //wi = weight_block[2*p+1];
//    //out_r = out_r + (pr*wr - pi*wi);
//    //out_i = out_i + (pr*wi + pi*wr);
//    out_r = out_r + (pr - pi);
//    out_i = out_i + (pr + pi);
//  }
//  out_block[0] = out_r;
//  out_block[1] = out_i;
//}

void beamform_cpu(float *weights, char *in, float *out, int npols, int nchans, int nbeams)
{
  //int nbeams = blockDim.y;
  int b;
  int c;
  int t;
  
  int p;
  float out_r[NBEAMS];
  float out_i[NBEAMS];
  char pr, pi;
  float wr, wi;
  for(t=0; t<NTIMES; t++){
    for(c=0; c<NCHANS; c++){
      char *in_block = in + 2*(t*nchans*npols + c*npols);
      for (p=0; p<npols; p++){
        pr = in_block[2*p];
        pi = in_block[2*p+1];
        for (b=0; b<nbeams; b++) {
          float *weight_block = weights + 2*(c*npols*nbeams + b*npols);
          //wr = weight_block[2*p];
          //wi = weight_block[2*p+1];
          //out_r[b] = out_r[b] + (pr*wr - pi*wi);
          //out_i[b] = out_i[b] + (pr*wi + pi*wr);
          out_r[b] = out_r[b] + (pr - pi);
          out_i[b] = out_i[b] + (pr + pi);
        }
      }
      for (b=0; b<nbeams; b++) {
        float *out_block = out + 2*(c*NTIMES*nbeams + t*nbeams + b);
        out_block[0] = out_r[b];
        out_block[1] = out_i[b];
      }
    }
  }
}

__global__
void beamform(float *weights, char *in, float *out, int npols, int nchans, int nbeams)
{
  int ntimes = blockDim.x;
  //int nbeams = blockDim.y;
  int t = blockIdx.x;
  int b;
  int c = threadIdx.x;
  
  int p;
  float out_r[NBEAMS];
  float out_i[NBEAMS];
  char pr, pi;
  float wr, wi;
  char *in_block = in + 2*(t*nchans*npols + c*npols);
  for (p=0; p<npols; p++){
    pr = in_block[2*p];
    pi = in_block[2*p+1];
    for (b=0; b<nbeams; b++) {
      float *weight_block = weights + 2*(c*npols*nbeams + b*npols);
      //wr = weight_block[2*p];
      //wi = weight_block[2*p+1];
      //out_r[b] = out_r[b] + (pr*wr - pi*wi);
      //out_i[b] = out_i[b] + (pr*wi + pi*wr);
      out_r[b] = out_r[b] + (pr - pi);
      out_i[b] = out_i[b] + (pr + pi);
    }
  }
  for (b=0; b<nbeams; b++) {
    float *out_block = out + 2*(c*ntimes*nbeams + t*nbeams + b);
    out_block[0] = out_r[b];
    out_block[1] = out_i[b];
  }
}


#define GPUDEV 1
int main() {
  float *weights_d;
  float *in_d;
  float *out_d;
  float *weights_h;
  float *in_h;
  float *out_h;
  struct timespec start, stop;
  long long unsigned elapsed_ns;
  long long unsigned bytes;
  double gbps;

  bytes = NPOLS*NCHANS*NTIMES;


  fprintf(stdout, "Using device: %d\n", GPUDEV);
  gpuErrchk( cudaSetDevice(GPUDEV) );

  fprintf(stdout, "bytes processed (4-bit input): %llu\n", bytes);

  fprintf(stdout, "Malloc-ing\n");
  weights_h = (float *)malloc(NPOLS * NCHANS * NBEAMS * 2 * sizeof(float));
  in_h      = (float *)malloc(NPOLS * NCHANS * NTIMES * 2 * sizeof(char));
  out_h     = (float *)malloc(NTIMES* NCHANS * NBEAMS * 2 * sizeof(float));
  gpuErrchk( cudaMalloc(&weights_d, NPOLS * NCHANS * NBEAMS * 2 * sizeof(float)) );
  gpuErrchk( cudaMalloc(&in_d,      NPOLS * NCHANS * NTIMES * 2 * sizeof(char)) );
  gpuErrchk( cudaMalloc(&out_d,     NTIMES* NCHANS * NBEAMS * 2 * sizeof(float)) );
  gpuErrchk( cudaMemcpy(weights_d, weights_h, NPOLS * NCHANS * NBEAMS * 2 * sizeof(float), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(in_d, in_h, NPOLS * NCHANS * NTIMES * 2 * sizeof(char), cudaMemcpyHostToDevice) );

  dim3 blockGrid(NTIMES, NBEAMS);
  dim3 threadGrid(NCHANS);

  int n;
  fprintf(stdout, "Calling kernel\n");
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(n=0; n<8; n++){
    //beamform<<<blockGrid, threadGrid>>>(weights_d, (char *)in_d, out_d, NPOLS, NCHANS);
    beamform<<<NTIMES, NCHANS>>>(weights_d, (char *)in_d, out_d, NPOLS, NCHANS, NBEAMS);
    cudaDeviceSynchronize();
    //beamform_cpu(weights_h, (char *)in_h, out_h, NPOLS, NCHANS, NBEAMS);
  }
  clock_gettime(CLOCK_MONOTONIC, &stop);
  gpuErrchk( cudaMemcpy(out_h, out_d, NTIMES* NCHANS * NBEAMS * 2 * sizeof(float), cudaMemcpyDeviceToHost) );

  elapsed_ns = ELAPSED_NS(start, stop);
  gbps = 8*n*bytes / (float)elapsed_ns;

  fprintf(stdout, "Elapsed: %llu ns for %d beams\n", elapsed_ns, NBEAMS);
  fprintf(stdout, "Gbps: %g gbps\n", gbps);
  fprintf(stdout, "Bandwidth: %.2f MHz\n", 1000*gbps / 8 / NPOLS);
  
  cudaFree(weights_d);
  cudaFree(in_d);
  cudaFree(out_d);
  free(weights_h);
  free(in_h);
  free(out_h);

  return 0;
}
