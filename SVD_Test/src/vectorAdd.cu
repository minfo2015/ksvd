/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include "singular_value_decomposition.h"
#include <time.h>

//#include "cusolverRf.h"
#include <cusolverDn.h>

 using namespace std;

#include <fstream>
#include <sstream>


#define TEST_PASSED  0
#define TEST_FAILED  1




//for svd
#define M 200
#define N 100


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{

	/**** CPU SVD ****/
     double A[M][N];
     double U[M][N];
     double V[N][N];
     double singular_values[N];
     double* dummy_array;
     int err  = 1 ;

     //(your code to initialize the matrix A)
     //double *h_A = (double *)malloc(Nrows * Ncols * sizeof(double));
     //A = (double*) malloc(N * sizeof(double));
     for(int j = 0; j < M; j++)
    	 for(int i = 0; i < N; i++)
    	 {
    		 double val = (i + j*j) * sqrt((double)(i + j));
    		 //printf("Valor :  %f \n" , val);
    		 A[j][i] = val ;
    	 }

     dummy_array = (double*) malloc(N * sizeof(double));
     if (dummy_array == NULL) {printf(" No memory available\n"); exit(0); }

     printf("Matrices Creadas  \n");

     clock_t launch = clock();

     err = Singular_Value_Decomposition((double*) A, M, N, (double*) U, singular_values, (double*) V, dummy_array);

     clock_t done = clock();
     double diff_time = (done-launch);/// CLOCKS_PER_SEC ;

     printf("Singular Value Completed in :  %f \n" , diff_time);

     free(dummy_array);
     if (err < 0) printf(" Failed to converge\n");
     else { printf(" The singular value decomposition of A is \n");}




    printf("Done\n");
    return 0;
}

