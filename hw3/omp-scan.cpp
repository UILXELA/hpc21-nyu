#include <algorithm>
#include <stdio.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
	if (n==0) return;
	prefix_sum[0] = 0;
	int num_threads = omp_get_max_threads();
	//printf("%d",num_threads);
	omp_set_num_threads(num_threads);
	long block_size = n/num_threads+1;
	long end_ind[num_threads];
	long offset[num_threads];
	offset[0]=0;
	#pragma omp parallel for schedule(static,block_size)
	for (long i = 1; i < n; i++) {
  	prefix_sum[i] = prefix_sum[i-1] + A[i-1];
	}

	for (int i = 1; i<num_threads; i++){
		offset[i]=offset[i-1]+prefix_sum[i*block_size];
	}
	#pragma omp parallel for schedule(static,block_size)
	for (long i = 1; i < n; i++) {
  	prefix_sum[i] = prefix_sum[i] + offset[omp_get_thread_num()];
	}
}


int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
