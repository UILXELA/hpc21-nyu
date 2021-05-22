#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void vec_prod(double* sum_ptr, const double* a, const double* b, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i]*b[i];
  *sum_ptr = sum;
}

void mat_prod(double** vec_ptr, const double* mat, const double* vec, long M, long N){
  #pragma omp parallel for schedule(static)
  for (long j = 0; j < M; j++){
    (*vec_ptr)[j] = 0;
    for (long i = 0; i < N; i++){ 
      (*vec_ptr)[j] += mat[M*j+i]*vec[i];
    }
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

__global__ void vecProd_kernel(double* sum, const double* a, const double* b,long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx]*b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void reduction_kernel2(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void matVectMul_kernel(double* mat, double* vec, double* result, long M, long N) {
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    double accum;
    accum = 0;

    if (row < M) {
        for (int i = 0; i < N; i++) {
            accum += mat[row * N + i] * vec[i];
        }
      result[row] = accum;
    }
}

int main() {
  printf("\n======= Vector Inner Product ========\n");
  long N = (1UL<<25);
  double *x;
  double *y;
  
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  
 
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++){
    x[i] = 1.0/(i+1);
    y[i] = 1.0/(i+1);
  }

  double sum_ref, sum;
  double tt = omp_get_wtime();
  vec_prod(&sum_ref, x, y, N);
  printf("CPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *y_d, *z_d, *mat_d, *vec_result_d; //mat:M*N
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
  
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&z_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks

  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  tt = omp_get_wtime();


  double* sum_d = z_d;
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  vecProd_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, x_d, y_d, N);
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d + N, sum_d, N);
    sum_d += N;
  }


  cudaMemcpyAsync(&sum, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("Error = %f\n", fabs(sum-sum_ref));

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFreeHost(x);
  cudaFreeHost(y);


  printf("\n======= Matrix-Vector Multiplication ========\n");

  long M = (1UL<<10);
  N = (1UL<<10);

  cudaMallocHost((void**)&x, N * sizeof(double));
 
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++){
    x[i] = 1.0/(i+1);
  }

  cudaMalloc(&x_d, N*sizeof(double));
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);

  double *mat = new double[M * N];
 
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < M*N; i++){
    mat[i] = 1.0/(i+1);
  }

  double* vec_result_ref  = new double[M];
  double * vec_result = new double[M];
  tt = omp_get_wtime();
  mat_prod(&vec_result_ref, mat, x, M, N);
  printf("CPU Bandwidth = %f GB/s\n", M*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  cudaMalloc(&vec_result_d, M*sizeof(double));
  cudaMalloc(&mat_d, M*N*sizeof(double));
  cudaMemcpyAsync(mat_d, mat, M*N*sizeof(double), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  tt = omp_get_wtime();

  long block = 512;
  long grid = ceil(double(M)/double(block));

  matVectMul_kernel<<<grid,block>>>(mat_d, x_d, vec_result_d, M, N);

  cudaMemcpyAsync((void*) vec_result, (void*) vec_result_d, M*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", M*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double vec_err = 0;

  #pragma omp parallel for schedule(static) reduction(+:vec_err)
  for (long i=0;i<M;i++){
    vec_err += fabs(vec_result[0]-vec_result_ref[0]);
  }

  printf("Error = %f\n", vec_err);


  cudaFree(x_d);
  cudaFree(mat_d);
  cudaFree(vec_result_d);
  cudaFreeHost(x);
  delete [] mat;
  delete [] (vec_result_ref);
  delete [] vec_result;


  return 0;
}

