# Final project - Partitioning of large point clouds using an adaptive octree

The project is very experimental. Directly modify the source code to try different things. 

## To run the project, you need:

  A point cloud in .bin format. The provided one is a small one fused from point clouds from the KITTI Dataset. If you want to test the program with it, please change the point cloud size as described below. Or even though it may not break anything, use it at your own risk. 
  
  The p4est library, which can be found at: https://github.com/cburstedde/p4est
  
  MPI and OpenMP
  
  
Before compiling, change the the line containing the file path

```C++
MPI_File_open(MPI_COMM_WORLD, "./test.bin",MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
```

and the SIZE defined explicitly in source by

```C++
define SIZE 18833843
```

To compile:
```bash
mpicxx -I/PATH_TO_P4EST/include -L/PATH_TO_P4EST/lib XXX.cpp -lp4est -lsc -lz -std=c++11 -O3 -march=native [-fopenmp] -o XXX
```

## What I learned from the project
There are three algorithms implemented. 'bin_iter.cpp' can only iterate thrugh newly created bins/octants with the help of a sorted array of all points encoded by Morton Order. 'mem_heavy.cpp' binds a separate container to every octant, therefore memory heavy. 'point_iter.cpp' iterates through all points and assign them to the right bins. It suffers from too many iterations for always. 


In terms of performance: bin_iter.cpp > mem_heavy.cpp > point_iter.cpp.

In terms of parallelism: bin_iter.cpp > mem_heavy.cpp > point_iter.cpp.

'bin_iter.cpp' uses a parallel sorting algorithm and can parallellize the iteraions. It is hard for 'mem_heavy.cpp' to benefit from parallelism because of the memory bound.'point_iter.cpp' benefits a lot from parallelized iteration but it is too slow.

MPI was tried by having processes to split the file and start their own octree. Results are very bad. The skewness of the octree makes balancing workload very difficult. The number of bins for each process is only slightly smaller than the single process case. Plus, reduction in the number of points benifits the 'point_iter' algorithm only. To scale up using MPI, all processes need to reach consensus on the point count of every octant. For 1M bins, that is 8MB data each process. The data needs to be broadcasted, which is an unacceptable cost to have. 

# Thanks
Thanks Professor Stadler for actively supporting.
Thanks to the author of https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/ for the morton encoding algorithms.
