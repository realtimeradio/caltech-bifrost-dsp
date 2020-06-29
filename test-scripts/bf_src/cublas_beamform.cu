#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define NPOLS 1
#define NANTS (2*352)
#define NCHANS 192
#define NTIMES 480
#define NTIMES_SUM 24
#define NBEAMS 32

#define NTIMEBLOCKS (NTIMES / NTIMES_SUM)

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

__constant__ unsigned int lut[16] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0};

// Transpose time x chan x pol x 4+4 bit to
// chan x pol x time x 32+32 bit float
__global__ void trans_4bit_to_float(unsigned char *in,
                                    float *out,
                                    int n_pol,
                                    int n_chan,
                                    int n_time
                                   ) {
  //long long int tid = blockDim.y*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
  //int pol  = tid % n_pol;
  //int chan = (tid / n_pol) % n_chan;
  //int time = (tid / (n_pol * n_chan));
  int time = blockIdx.x;
  int chan = blockIdx.y;
  int pol = threadIdx.x;
  long long int old_index = time*n_chan*n_pol + chan*n_pol + pol;
  long long int new_index = chan*n_pol*n_time + pol*n_time + time;
  float real, imag;
  real = lut[in[old_index] >> 4];
  imag = lut[in[old_index] & 0b1111];
  out[2*new_index] = real;
  out[2*new_index+1] = imag;
}

// Transpose chan x beam x pol x time x 32+32 float to
// beam x time[part-summed] x chan x [XX,YY,XY*_r,XY*_i] x 32 float
// Each thread deals with two pols of a beam, and sums over n_time_sum time samples
__global__ void trans_output_and_sum(float *in,
                                    float *out,
                                    int n_chan,
                                    int n_beam,
                                    int n_time,
                                    int n_time_sum
                                   ) {
  //long long int tid = blockDim.y*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
  //int pol  = tid % n_pol;
  //int chan = (tid / n_pol) % n_chan;
  //int time = (tid / (n_pol * n_chan));
  int chan = blockIdx.x;
  int beam = blockIdx.y;
  int time = threadIdx.x;
  long long int old_index = chan*n_beam*n_time*2 + beam*n_time*2 + time*n_time_sum; // start index for n_time/n_time_sum samples
  long long int new_index = beam*(n_time / n_time_sum)*n_chan + time*n_chan + chan;
  float xx=0., yy=0., xy_r=0., xy_i=0.;
  float x_r, x_i, y_r, y_i;
  int t;
  for (t=0; t<n_time_sum; t++) {
    x_r = in[2*old_index + 2*t];
    x_i = in[2*old_index + 2*t + 1];
    y_r = in[2*old_index + n_time + 2*t];
    y_i = in[2*old_index + n_time + 2*t + 1];
    xx = xx + x_r*x_r + x_i*x_i;
    yy = yy + y_r*y_r + y_i*y_i;
    xy_r = xy_r + x_r*y_r + x_i*y_i;
    xy_i = xy_i + x_i*y_r - x_r*y_i;
  }
  out[4*new_index] = xx;
  out[4*new_index+1] = yy;
  out[4*new_index+2] = xy_r;
  out[4*new_index+3] = xy_i;
}

__global__ void complex2pow(float *in, float *out, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    out[tid] = (in[2*tid]*in[2*tid] + in[2*tid + 1]*in[2*tid + 1]);
  }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
Transa	CUDA_OP_N	Matrix A (Fourier coefficient matrix) is not transposed
Transb	CUDA_OP_N	Matrix B (input data) is not transposed
M	N_BEAMS	Number of rows of A/C
N	N_TIMESTEPS_PER_CALL	Number of columns of B/C
K	N_ANTENNAS	Number of columns of A, number of rows in B
Alpha	1.0/127	Fourier coefficients are 8-bit ints so must be normalized to 1
Atype	CUDA_C_8I	Data type of Fourier coefficient matrix (i.e. 8-bit + 8-bit integers)
Lda	N_BEAMS	Leading dimension of Fourier coefficient matrix
strideA	N_BEAMS*N_ANTENNAS	Stride between different frequencies in A
Btype	CUDA_C_8I	Data type of input data matrix
Ldb	N_ANTENNAS	Leading dimension of input matrix
StrideB	N_ANTENNAS*N_TIMESTEPS_PER_CALL	Stride between different frequencies in input matrix
Beta	0	Zero out the output data tensor
Ctype	CUDA_C_32F	Data type of output matrix
Ldc	N_BEAMS	Leading Dimension of output matrix
strideC	N_BEAMS*N_TIMESTEPS_PER_CALL	Stride between different frequencies in output matrix
batchCount	NCHANS	How many frequencies
computeType	CUDA_C_32F	Internal datatype
Algo	CUBLAS_GEMM_DEFAULT_TENSOR_OP	Use tensor operations
*/


//gpuBLASchk(cublasGemmStridedBatchedEx(handle[st], CUBLAS_OP_N, CUBLAS_OP_N,
//                                                        fourier_coefficients_rows, B_cols, fourier_coefficients_cols,
//                                                        d_inv_max_value,
//                                                        d_fourier_coefficients, CUDA_C_8I, fourier_coefficients_rows, fourier_coefficients_stride,
//                                                        &d_B[N_CX_IN_PER_GEMM*st], CUDA_C_8I, B_rows, B_stride,
//                                                        d_zero,
//                                                        &d_C[N_CX_OUT_PER_GEMM*st], CUDA_C_32F, C_rows, C_stride,
//                                                        NCHANS, CUDA_C_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

//cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
//                            cublasOperation_t transa,
//                            cublasOperation_t transb,
//                            int m,
//                            int n,
//                            int k,
//                            const void    *alpha,
//                            const void     *A,
//                            cudaDataType Atype,
//                            int lda,
//                            long long int strideA,
//                            const void     *B,
//                            cudaDataType Btype,
//                            int ldb,
//                            long long int strideB,
//                            const void    *beta,
//                            void           *C,
//                            cudaDataType Ctype,
//                            int ldc,
//                            long long int strideC,
//                            int batchCount,
//                            cudaDataType computeType,
//                            cublasGemmAlgo_t algo)


/* Error checking for cuBLAS */
void gpuBLASchk(int errval) {
	if (errval != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Failed BLAS call, error code %d\n", errval);
	}
}




#define GPUDEV 1
#define LOOPCNT 20
int main() {
  float *weights_d;
  unsigned char  *in4_d;
  float *in32_d;
  float *out_d;
  float *pow_d;
  float *sum_out_d;
  float *weights_h;
  unsigned char  *in4_h;
  float *out_h;
  struct timespec start, stop;
  long long unsigned elapsed_ns;
  long long unsigned bytes;
  double gbps;
  float alpha = 1.0;
  float beta = 0.0;

  bytes = NANTS*NPOLS*NCHANS*NTIMES;


  fprintf(stdout, "Using device: %d\n", GPUDEV);
  gpuErrchk( cudaSetDevice(GPUDEV) );

  
  cublasHandle_t handle;
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreate(&(stream)));
  gpuBLASchk(cublasCreate(&(handle)));
  gpuBLASchk(cublasSetStream(handle, stream));
  gpuBLASchk(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  //gpuBLASchk(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  gpuBLASchk(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  //cublasHandle_t handle[LOOPCNT];
  //cudaStream_t stream[LOOPCNT];
  //int i;
  //for (i=0; i<LOOPCNT; i++) {
  //  fprintf(stdout, "Initializing handle %d\n", i);
  //  gpuErrchk(cudaStreamCreate(&(stream[i])));
  //  gpuBLASchk(cublasCreate(&(handle[i])));
  //  gpuBLASchk(cublasSetStream(handle[i], stream[i]));
  //  gpuBLASchk(cublasSetPointerMode(handle[i], CUBLAS_POINTER_MODE_HOST));
  //  //gpuBLASchk(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  //  gpuBLASchk(cublasSetMathMode(handle[i], CUBLAS_TENSOR_OP_MATH));
  //}

  fprintf(stdout, "bytes processed (4-bit input): %llu\n", bytes);

  fprintf(stdout, "Malloc-ing\n");
  weights_h = (float *)malloc(NANTS * NPOLS * NCHANS * NBEAMS * 2 * sizeof(float));
  in4_h      = (unsigned char *)malloc(NANTS * NPOLS * NCHANS * NTIMES * sizeof(char));
  out_h     = (float *)malloc(NTIMES* NCHANS * NBEAMS * 2 * sizeof(float));
  gpuErrchk( cudaMalloc(&weights_d, NANTS * NPOLS * NCHANS * NBEAMS * 2 * sizeof(float)) );
  gpuErrchk( cudaMalloc(&in4_d,      NANTS * NPOLS * NCHANS * NTIMES * sizeof(char)) );
  gpuErrchk( cudaMalloc(&in32_d,      NANTS * NPOLS * NCHANS * NTIMES * 2 * sizeof(float)) );
  gpuErrchk( cudaMalloc(&out_d,     NTIMES* NCHANS * NBEAMS * 2 * sizeof(float)) );
  //gpuErrchk( cudaMalloc(&pow_d,     NTIMES* NCHANS * NBEAMS * sizeof(float)) );
  gpuErrchk( cudaMalloc(&sum_out_d,     NTIMEBLOCKS * NCHANS * NBEAMS/2 * 4 * sizeof(float)) );
  gpuErrchk( cudaMemcpy(weights_d, weights_h, NANTS * NPOLS * NCHANS * NBEAMS * 2 * sizeof(float), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(in4_d, in4_h, NANTS * NPOLS * NCHANS * NTIMES * sizeof(char), cudaMemcpyHostToDevice) );

  dim3 transBlockGrid(NTIMES, NCHANS);
  dim3 transThreadGrid(NPOLS);
  dim3 sumBlockGrid(NCHANS, NBEAMS/2);
  dim3 sumThreadGrid(NTIMES / NTIMES_SUM);

  int n;
  fprintf(stdout, "Calling kernel\n");
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(n=0; n<LOOPCNT; n++){
    // Transpose input data and promote to float.
    // CUBLAS doesn't support float coeffs with int8 data
    trans_4bit_to_float<<<transBlockGrid, transThreadGrid, 0, stream>>>(in4_d, in32_d, NPOLS, NCHANS, NTIMES);
    cudaStreamSynchronize(stream);

    // GEMM:
    // C <= alpha*AB + beta*C
    // alpha = 1.0
    // beta = 0.0
    // A matrix: beamforming coeffs (NBEAMS * NANTS)
    // B matrix: data matrix (NANTS * NTIMES)
    gpuBLASchk(cublasGemmStridedBatchedEx(handle,
      CUBLAS_OP_N, // transpose A?
      CUBLAS_OP_N, // transpose B?
      NBEAMS,      // m
      NTIMES,      // n
      NANTS,       // k
      // Coeffs
      &alpha,      // alpha
      weights_d,   // A
      CUDA_C_32F,  // A type
      //CUDA_C_8I,  // A type
      NBEAMS,      // Lda
      NBEAMS*NANTS,// strideA : stride size
      // Data
      in32_d,        // B
      CUDA_C_32F,   // B type
      //CUDA_C_8I,   // B type
      NANTS,       // Ldb
      NANTS*NTIMES,// strideB : stride size
      &beta,       // beta
      // Results
      out_d,       // C
      CUDA_C_32F,  // Ctype 
      NBEAMS,      // Ldc
      NBEAMS*NTIMES, // Stride C
      NCHANS, // batchCount
      CUDA_C_32F,  // compute type
      CUBLAS_GEMM_DEFAULT_TENSOR_OP // algo
      ));
    cudaStreamSynchronize(stream);
    trans_output_and_sum<<<sumBlockGrid, sumThreadGrid, 0, stream>>>(out_d, sum_out_d, NCHANS, NBEAMS/2, NTIMES, NTIMES_SUM);
    cudaStreamSynchronize(stream);

  }
  clock_gettime(CLOCK_MONOTONIC, &stop);
  //gpuErrchk( cudaMemcpy(out_h, out_d, NTIMES* NCHANS * NBEAMS * 2 * sizeof(float), cudaMemcpyDeviceToHost) );

  elapsed_ns = ELAPSED_NS(start, stop);
  gbps = 8*LOOPCNT*bytes / (float)elapsed_ns;

  fprintf(stdout, "Elapsed: %llu ns for %d beams\n", elapsed_ns, NBEAMS);
  fprintf(stdout, "Gbps: %g gbps\n", gbps);
  fprintf(stdout, "Bandwidth: %.2f MHz\n", 1000*gbps / 8 / NPOLS / NANTS);
  
  cudaFree(weights_d);
  cudaFree(in4_d);
  cudaFree(in32_d);
  cudaFree(out_d);
  free(weights_h);
  free(in4_h);
  free(out_h);

  return 0;
}
