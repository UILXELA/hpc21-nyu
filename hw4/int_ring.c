#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


//int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
int main(int argc, char *argv[]) {
	if (argc!=2){
		printf("[ERR] Correct usage: 'int_ring N");
		return -1;
	}

	int rank, N, size, source, dest, i;
	unsigned long msg;
	MPI_Status status;

  	MPI_Init(&argc, &argv);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
  	int name_len;
  	MPI_Get_processor_name(processor_name, &name_len);
  	printf("Rank %d/%d running on %s.\n", rank, size, processor_name);

	N = atoi(argv[1]);
	source = (rank-1+size)%size;	//the rank to receive msg from, +size to wrap around
	dest = (rank+1)%size;	//the rank to send the msg to

	if (rank==0) msg=0;

	MPI_Barrier(MPI_COMM_WORLD);
  	double tt = MPI_Wtime();

	for (i=0;i<N;i++){
		if(rank==0){
			MPI_Send(&msg, 1, MPI_LONG, dest, 1, MPI_COMM_WORLD);
			MPI_Recv(&msg, 1, MPI_LONG, source, 1, MPI_COMM_WORLD,&status);
		}else{
			MPI_Recv(&msg, 1, MPI_LONG, source, 1, MPI_COMM_WORLD,&status);
			msg+=rank;
			MPI_Send(&msg, 1, MPI_LONG, dest, 1, MPI_COMM_WORLD);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
  	double elapsed = MPI_Wtime() - tt;

	//if(rank==0){
	//	MPI_Recv(&msg, 1, MPI_LONG, source, 1, MPI_COMM_WORLD,&status);
	//	printf("The final message is %lu\n",msg);
	//}

  	if (rank==0) {
		printf("The final message is %lu\n",msg);
		double latency = (double)1000*elapsed/N/size;
    	printf("Time elapsed is %f s for %d loops among %d processes\n", elapsed, N, size);
		printf("Latency is %f ms.\n", latency);
  	}

	MPI_Finalize();

	return 0;

}


