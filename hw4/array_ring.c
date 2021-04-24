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
    int arr_length = (int)2e6 / sizeof(long);
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
    long* msg = (long*) calloc(sizeof(long),arr_length);	//initialize all to 0
    
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    for (i=0;i<N;i++){
        if(rank==0){	
            MPI_Send(msg, arr_length, MPI_LONG, dest, 1, MPI_COMM_WORLD);
            MPI_Recv(msg, arr_length, MPI_LONG, source, 1, MPI_COMM_WORLD,&status);
        }else{
            MPI_Recv(msg, arr_length, MPI_LONG, source, 1, MPI_COMM_WORLD,&status);
            //msg[0]+=rank;
            MPI_Send(msg, arr_length, MPI_LONG, dest, 1, MPI_COMM_WORLD);
        }
    }

    free(msg);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;

    if (rank==0) {
        double bandwidth = (double)2.0*N*size/elapsed/1000;	//fixed size: 2MB
        printf("Time elapsed is %f s for %d loops among %d processes\n", elapsed, N, size);
        printf("Bandwidth is %f GB/s.\n", bandwidth);
    }

    MPI_Finalize();

    return 0;

}


