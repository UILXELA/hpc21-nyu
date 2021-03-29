#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <math.h> 
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

int ITERATION=100000;

using namespace std;
double residual(int N, double a, double b, double* u, double* f ){
	double residual_squared=0;
	double f_hat;
	for(int j=1;j<=N;j++){
		for(int k=1;k<=N;k++){
			f_hat = a*u[j*N+k] + b*u[j*N+k-1] + b*u[j*N+k+1] + b*u[(j-1)*N+k] + b*u[(j+1)*N+k];
			residual_squared += pow(f_hat-f[(j-1)*N+k-1],2.0);
			//printf("%d,%d: %f\n",j,k,f_hat);
		}
	}
	return sqrt(residual_squared);
}

void jacobi(int N, double* next_u, double* u, double* f ){
	double h_squared=pow(1.0/(N+1),2.0);
	for(int i=0;i<ITERATION;i++){
		for(int j=1;j<=N;j++){
			for(int k=1;k<=N;k++){
				next_u[j*N+k] = (h_squared*f[(j-1)*N+k-1]+u[j*N+k-1] +u[j*N+k+1] + u[(j-1)*N+k] + u[(j+1)*N+k])/4.0;
			}
		}
		//swapping pointers as demonstrated in class faster than memcpy
		double* temp_u = u;
    	u = next_u;
    	next_u = temp_u;
	}
	return;
}

void jacobi_omp(int N, double* next_u, double* u, double* f ){
	double h_squared=pow(1.0/(N+1),2.0);
	for(int i=0;i<ITERATION;i++){
		#pragma omp parallel for schedule(static) collapse(2)
		for(int j=1;j<=N;j++){
			for(int k=1;k<=N;k++){
				next_u[j*N+k] = (h_squared*f[(j-1)*N+k-1]+u[j*N+k-1] +u[j*N+k+1] + u[(j-1)*N+k] + u[(j+1)*N+k])/4.0;
			}
		}
		double* temp_u = u;
    	u = next_u;
    	next_u = temp_u;
	}
	return;
}


int main(int argc, char** argv){
	if(argc!=2){
		cout<<"No input for N"<<endl;
		return 1;
	}
	Timer t;
	int N=atoi(argv[1]);


	//No OMP
	double* u = (double*) calloc((N+2)*(N+2), sizeof(double));
	double* f = (double*) calloc(N*N, sizeof(double));
	double* u_copy = (double*) calloc((N+2)*(N+2), sizeof(double));

	//Initialize a,b,u,f
	double h_squared=pow(1.0/(N+1),2.0);
	double a=4.0/h_squared;
	double b=-1.0/h_squared;
	for(int i =0;i<N*N;i++){
		f[i]=1;
	}

	//if no omp, just one thread
	int num_threads = 1;
	#ifdef _OPENMP
		num_threads = omp_get_max_threads();
	#endif

	printf("\nJacobi with %d omp threads (N=%d):\n",num_threads,N);
	t.tic();
	jacobi_omp(N,u_copy,u,f);
	//jacobi(N,u_copy,u,f);
	double j_t=t.toc();
	double j_res=residual(N,a,b,u,f);	
	
	printf("Time: %.4fs, Residual: %f\n",j_t,j_res);

	free(u);
	free(f);
	free(u_copy);


return 0;
}
