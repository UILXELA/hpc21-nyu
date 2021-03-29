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

void gauss_seidel(int N, double* u, double* f ){
	double h_squared=pow(1.0/(N+1),2.0);
	for(int i=0;i<ITERATION;i++){
		for(int j=1;j<=N;j++){
			for(int k=1;k<=N;k++){
				u[j*N+k] = (h_squared*f[(j-1)*N+k-1]+u[j*N+k-1] +u[j*N+k+1] + u[(j-1)*N+k] + u[(j+1)*N+k])/4.0;
			}
		}
	}
	return;
}


void gauss_seidel_omp(int N, double* u, double* f ){
	double h_squared=pow(1.0/(N+1),2.0);
	for(int i=0;i<ITERATION;i++){
		#pragma omp parallel for schedule(static) collapse(2)
		for(int j=1;j<=N;j++){
			for(int k=1;k<=N;k+=2){
				int k2=(j%2!=0) ? k+1 :k;
				if(k2>N) continue;
				u[j*N+k2] = (h_squared*f[(j-1)*N+k2-1]+u[j*N+k2-1] +u[j*N+k2+1] + u[(j-1)*N+k2] + u[(j+1)*N+k2])/4.0;
			}
		}
		#pragma omp parallel for schedule(static) collapse(2)
		for(int j=1;j<=N;j++){
			for(int k=1;k<=N;k+=2){
				int k2=(j%2==0) ? k+1 :k;
				if(k2>N) continue;
				u[j*N+k2] = (h_squared*f[(j-1)*N+k2-1]+u[j*N+k2-1] +u[j*N+k2+1] + u[(j-1)*N+k2] + u[(j+1)*N+k2])/4.0;
			}
		}
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


	double* u = (double*) calloc((N+2)*(N+2), sizeof(double));
	double* f = (double*) malloc(N*N*sizeof(double));

	//Initialize a,b,u,f. a,b are coeffs
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

	printf("G-S with %d omp threads (N=%d):\n",num_threads,N);
	t.tic();
	gauss_seidel_omp(N,u,f);
	//gauss_seidel(N,u,f);
	double gs_t=t.toc();
	double gs_res=residual(N,a,b,u,f);	
	
	
	printf("Time: %.4fs, Residual: %f\n",gs_t,gs_res);

	free(u);
	free(f);

return 0;
}
