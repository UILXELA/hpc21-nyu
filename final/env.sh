#mpicxx -I/home/zl3768/p4est_debug/include -L/home/zl3768/p4est_debug/lib test.cpp -lp4est -lsc -lz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zl3768/p4est_nompi/lib/
module load mpi/openmpi-x86_64
module load gcc-7.4