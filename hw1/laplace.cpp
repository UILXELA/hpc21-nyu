#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <math.h> 
#include <iostream>

long RESIDUAL_FACTOR=1e6;
int ITERATION=100;

using namespace std;
double residual(int N, double* A, double* u, double* f ){
	double residual_squared=0;
	for (int i=0;i<N;i++){
		double Au_i=0;
		for (int j=0;j<N;j++){
			Au_i += A[i*N+j]*u[j];
			//cout<<Au_i<<endl;
		}
		residual_squared += pow(Au_i-f[i],2.0);
	}
	return sqrt(residual_squared);
}

void jacobi(int N, double* A, double* u, double* f ){
	double cur_residual = residual(N,A,u,f);
	double* u_orig=u;
	double next_u[N];
	double target_residual = cur_residual / RESIDUAL_FACTOR;
	int i = 0;
	while(i<ITERATION && (cur_residual>target_residual)){
	//for(int i=0;i<ITERATION;i++){
		for(int j=0;j<N;j++){
			double sum = -A[j*N+j]*u[j];
			for(int k=0;k<N;k++){
				sum += A[j*N+k]*u[k];
			}
			next_u[j]=(f[j]-sum)/A[j*N+j];
		}
		u=next_u;
		cur_residual = residual(N,A,u,f);
		printf("%10d %10f\n",i,cur_residual);
		i++;
	}
	printf("Terminated at iteration %d.\nResidual: %f\n\n\n", --i,cur_residual);
	memcpy(u_orig,u,N*sizeof(double));
	return;
}

void gauss_seidel(int N, double* A, double* u, double* f ){
	double cur_residual = residual(N,A,u,f); 
	double target_residual = cur_residual / RESIDUAL_FACTOR;
	int i = 0;
	while(i<ITERATION && cur_residual>target_residual){
	//for(int i=0;i<ITERATION;i++){
		for(int j=0;j<N;j++){
			double sum = -A[j*N+j]*u[j];
			for(int k=0;k<N;k++){
				sum += A[j*N+k]*u[k];
			}
			u[j]=(f[j]-sum)/A[j*N+j];
		}
		cur_residual = residual(N,A,u,f);
		printf("%10d %10f\n",i,cur_residual);
		i++;
	}
	printf("Terminated at iteration %d.\nResidual: %f\n\n\n", --i,cur_residual);
	return;
}


int main(int argc, char** argv){
	if(argc!=2){
		cout<<"No input for N"<<endl;
		return 1;
	}
	int N=atoi(argv[1]);
	double* A = (double*) calloc(N*N,sizeof(double)); //row major
	double* u = (double*) malloc(N*sizeof(double));
	double* f = (double*) malloc(N*sizeof(double));
	double* u_copy = (double*) malloc(N*sizeof(double));

	//Initialize A,u,f
	//A
	double h_squared=pow(1.0/(N+1),2.0);
	double a=2.0/h_squared;
	double b=-1.0/h_squared;
	for(int i=1;i<N-1;i++){
		A[i*N+i]=a;
		A[i*N+i+1]=b;
		A[i*N+i-1]=b;
	}
	A[0]=a;
	A[1]=b;
	A[N*N-1]=a;
	A[N*N-2]=b;

	//u,f
	for(int i=0;i<N;i++){
		u[i]=0;
		f[i]=1;
	}

	Timer t;

	printf("Gauss-Seidel (N=%d):\n",N);
	memcpy(u_copy,u,N*sizeof(double));
	t.tic();
	gauss_seidel(N,A,u_copy,f);
	double gs_t=t.toc();
	double gs_res=residual(N,A,u_copy,f);
	

	printf("Jacobi (N=%d):\n",N);
	memcpy(u_copy,u,N*sizeof(double));
	t.tic();
	jacobi(N,A,u_copy,f);
	double j_t=t.toc();
	double j_res=residual(N,A,u_copy,f);

	/* Correctness check
	for(int i=0;i<N;i++){
		double check=0;
		for(int j=0;j<N;j++){
			check+=A[i*N+j]*u_copy[j];
		}
		cout<<check<<endl;
	}
	*/


	
	
	printf("Gauss-Seidel time: %.4fs, Residual: %f\n",gs_t,gs_res);
	printf("Jacobi time: %.4fs, Residual: %f\n",j_t,j_res);

	free(A);
	free(u);
	free(f);
	free(u_copy);


return 0;
}
