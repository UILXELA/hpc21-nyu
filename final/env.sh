#mpicxx -I/home/alex/p4est_nompi/include -L/home/alex/p4est_nompi/lib test.cpp -lp4est -lsc -lz -O3 -fopenmp
# mpirun --np 2 --map-by ppr:1:socket ./a.out
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alex/p4est_nompi/lib/
#module load mpi/openmpi-x86_64
#module load gcc-7.4
