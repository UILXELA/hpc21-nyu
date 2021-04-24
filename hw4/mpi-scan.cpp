#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <string.h>

/*int MPI_Allgather(const void *sendbuf, int  sendcount,
 *    MPI_Datatype sendtype, void *recvbuf, int recvcount,
 *    MPI_Datatype recvtype, MPI_Comm comm)
*/

/*MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
  *  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
  *  MPI_Comm comm)
*/

/*MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
  *  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
  *  MPI_Comm comm)
*/

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

int main(int argc, char *argv[]) {

  int rank, size;
  long N, i, n, j, offset;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, size, processor_name);

  N=48000000;
  n = N / size; //assuming N%size==0 

  long* data;
  long* seq_result;
  long* mpi_result;
  long* local_in = (long*) malloc(n * sizeof(long));
  long* local_out = (long*) malloc(n * sizeof(long));
  long* offset_arr = (long*) malloc(size * sizeof(long));
  double tt;
  double seq_time;

  if(rank==0){
    data = (long*) malloc(N * sizeof(long));
    seq_result = (long*) malloc(N * sizeof(long));
    mpi_result = (long*) calloc((N+1),sizeof(long));
    for(i=0;i<N;i++){
      data[i] = rand();
    }
    tt = MPI_Wtime();
    scan_seq(seq_result, data, N);
    seq_time = MPI_Wtime() - tt;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tt = MPI_Wtime();


  MPI_Scatter(data, n, MPI_LONG, local_in, n, MPI_LONG, 0, MPI_COMM_WORLD);

  local_out[0] = local_in[0];
  for (long i = 1; i < n; i++) {
    local_out[i] = local_out[i-1] + local_in[i];
  }

  //for(i=0;i<n;i++) std::cout<<local_out[i]<<",";
  //printf("\n");

  MPI_Allgather(local_out+n-1, 1, MPI_LONG, offset_arr, 1, MPI_LONG, MPI_COMM_WORLD);

  offset=0;
  for(i=0; i<rank;i++){
    offset += offset_arr[i];
  }

  for(i=0; i<n; i++){
    local_out[i] += offset;
  }

  //add the 0 to the beginning and omit the very last term
  MPI_Gather(local_out, n, MPI_LONG, &(mpi_result[1]), n, MPI_LONG, 0, MPI_COMM_WORLD);

  free(local_in);
  free(local_out);
  free(offset_arr);

  MPI_Barrier(MPI_COMM_WORLD);
  double mpi_time = MPI_Wtime() - tt;

  if(rank==0){
    long err = 0;
    for (long i = 0; i < N; i++) err = std::max(err, std::abs(seq_result[i] - mpi_result[i]));
    printf("error = %ld\n", err);
    printf("sequential-scan = %fs\n", seq_time);
    printf("parallel-scan   = %fs\n", mpi_time);

    free(seq_result);
    free(mpi_result);
    free(data);

  }

  MPI_Finalize();

  return 0;
}
