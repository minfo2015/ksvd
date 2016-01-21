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
#include <stdlib.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include<iostream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<math.h>

//#include "cusolverRf.h"
#include <cusolverDn.h>
#include "Utilities.cuh"

 using namespace std;


#include <fstream>
#include <sstream>

#include "addnoise_function.h"
#include "io_png.h"
#include "utilities.h"
#include "ksvd.h"


#define TEST_PASSED  0
#define TEST_FAILED  1


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

	 //! Check if there is the right call for the algorithm
		/**if (argc < 10)
		{
			cout << "usage: K-SVD image sigma noisy denoised difference bias diff_bias \
	                                                            doBias doSpeedUp" << endl;
			return EXIT_FAILURE;
		}*/

	/*************
		char *argv[] = {"","lena.png", "10", "ImNoisy.png", "ImDenoised.png", "ImDiff.png", "ImBias.png", "ImDiffBias.png", "0", "1" ,"0"} ;

		//! read input image
		cout << "Read input image...";
		size_t height, width, chnls;
		float *img = NULL;
		img = io_png_read_f32(argv[1], &width, &height, &chnls);
		//img = io_png_read_f32("", &width, &height, &chnls);
		if (!img)
		{
			cout << "error :: " << argv[1] << " not found  or not a correct png image" << endl;
			return EXIT_FAILURE;
		}
		cout << "done." << endl << endl;

		//! test if image is really a color image and exclude the alpha channel
		if (chnls > 2)
		{
		    unsigned k = 0;
		    while (k < width * height \
	            && img[k] == img[width * height + k] \
	            && img[k] == img[2 * width * height + k])
	            k++;
	        chnls = (k == width * height ? 1 : 3);
		}

		//! Printing some characterics of the input image
	    cout << "image size : " << endl;
	    cout << "-width    = " << width << endl;
	    cout << "-height   = " << height << endl;
	    cout << "-channels = " << chnls << endl << endl;

	    //! Variables initialization
		double fSigma   = atof(argv[2]);
		unsigned wh     = (unsigned) width * height;
		unsigned whc    = (unsigned) wh * chnls;

		//! Add noise
		cout << "Add noise [sigma = " << fSigma << "] ...";
		double *img_noisy    = new double[whc];
		float  *img_denoised = new float [whc];
		float  *img_bias     = new float [whc];

		for (unsigned c = 0; c < chnls; c++)
			fiAddNoise(&img[c * wh], &img_noisy[c * wh], fSigma, c, wh);
	    cout << "done." << endl;

		//! Denoising
		bool useAcceleration = atof(argv[9]);
		cout << "Applying K-SVD to the noisy image :" << endl;
	    ksvd_ipol((double) fSigma / 255.0l, img_noisy, img_denoised, width, height, chnls,
	              useAcceleration);
	    cout << endl;

	    if (atof(argv[8]))
	    {
	        cout << "Applying K-SVD to the original image :" << endl;
	        double *img_noisy_bias = new double[whc];
	        for (unsigned k = 0; k < whc; k++)
	            img_noisy_bias[k] = (double) img[k];
	        ksvd_ipol((double) fSigma / 255.0l, img_noisy_bias, img_bias, width, height,
	                  chnls, useAcceleration);
	        delete[] img_noisy_bias;
	        cout << endl;
	    }

	    //! Compute RMSE and PSNR
	    float rmse, rmse_bias, psnr, psnr_bias;
	    psnr_rmse(img, img_denoised, &psnr, &rmse, whc);
	    cout << endl;
	    cout << "For noisy image :" << endl;
	    cout << "PSNR: " << psnr << endl;
	    cout << "RMSE: " << rmse << endl << endl;
	    if (atof(argv[8]))
	    {
	        psnr_rmse(img, img_bias, &psnr_bias, &rmse_bias, whc);
	        cout << "For original image :" << endl;
	        cout << "PSNR: " << psnr_bias << endl;
	        cout << "RMSE: " << rmse_bias << endl << endl;
	    }

		//! writing measures
	    char path[13] = "measures.txt";
	    ofstream file(path, ios::out);
	    if(file)
	    {
	        file << "************" << endl;
	        file << "-sigma = " << fSigma << endl;
	        file << "-PSNR  = " << psnr << endl;
	        file << "-RMSE  = " << rmse << endl;
	        if (atof(argv[8]))
	        {
	            file << "-PSNR_bias  = " << psnr_bias << endl;
	            file << "-RMSE_bias  = " << rmse_bias << endl << endl;
	        }
	        file.close();
	    }
	    else
	        return EXIT_FAILURE;

		//! Compute Difference
		cout << "Compute difference...";
		fSigma *= 4.0f;
		float *img_diff      = new float[whc];
		float *img_diff_bias = new float[whc];
		float fValue, fValue_bias;

	    #pragma omp parallel for
	        for (unsigned k = 0; k < whc; k++)
	        {
	            fValue =  (img[k] - img_denoised[k] + fSigma) * 255.0f / (2.0f * fSigma);
	            fValue_bias =  (img[k] - img_bias[k] + fSigma) * 255.0f / (2.0f * fSigma);
	            img_diff[k] = (fValue < 0.0f ? 0.0f : (fValue > 255.0f ? 255.0f : fValue));
	            img_diff_bias[k] = (fValue_bias < 0.0f ? 0.0f : \
	                                (fValue_bias > 255.0f ? 255.0f : fValue_bias));
	        }
		cout << "done." << endl << endl;

		//! save noisy, denoised and differences images
		cout << "Save images...";
		float * img_noisy_f = new float[whc];
		for (unsigned k = 0; k < whc; k++)
	        img_noisy_f[k] = (float) (img_noisy[k] * 255.0l);

		if (io_png_write_f32(argv[3], img_noisy_f, width, height, chnls) != 0)
			cout << "... failed to save png image " << argv[3] << endl;

		if (io_png_write_f32(argv[4], img_denoised, width, height, chnls) != 0)
			cout << "... failed to save png image " << argv[4] << endl;

	    if (io_png_write_f32(argv[5], img_diff, width, height, chnls) != 0)
			cout << "... failed to save png image " << argv[5] << endl;
	    if (atof(argv[8]))
	    {
	        if (io_png_write_f32(argv[6], img_bias, width, height, chnls) != 0)
	            cout << "... failed to save png image " << argv[6] << endl;

	        if (io_png_write_f32(argv[7], img_diff_bias, width, height, chnls) != 0)
	            cout << "... failed to save png image " << argv[7] << endl;
	    }
	    cout << "done." << endl;

		//! Free Memory
		delete[] img_denoised;
		delete[] img_noisy;
		delete[] img_noisy_f;
		delete[] img_diff;
		delete[] img_bias;
		delete[] img_diff_bias;

		return EXIT_SUCCESS;


		********************************/

	/************** SVD DECOMPOSITION */



	// --- gesvd only supports Nrows >= Ncols
	// --- column major memory ordering

	const int Nrows = 700*3;
	const int Ncols = 500*3;


    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int *devInfo;
    gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // --- Setting the host, Nrows x Ncols matrix
    double *h_A = (double *)malloc(Nrows * Ncols * sizeof(double));
    for(int j = 0; j < Nrows; j++)
        for(int i = 0; i < Ncols; i++)
            h_A[j + i*Nrows] = (i + j*j) * sqrt((double)(i + j));

    printf("Matriz Creada");


    // --- Setting the device matrix and moving the host matrix to the device
    double *d_A;            gpuErrchk(cudaMalloc(&d_A,      Nrows * Ncols * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

    printf("Matriz en el Device");



    // --- host side SVD results space
    double *h_U = (double *)malloc(Nrows * Nrows     * sizeof(double));
    double *h_V = (double *)malloc(Ncols * Ncols     * sizeof(double));
    double *h_S = (double *)malloc(min(Nrows, Ncols) * sizeof(double));



    // --- device side SVD workspace and matrices
    double *d_U;            gpuErrchk(cudaMalloc(&d_U,  Nrows * Nrows     * sizeof(double)));
    double *d_V;            gpuErrchk(cudaMalloc(&d_V,  Ncols * Ncols     * sizeof(double)));
    double *d_S;            gpuErrchk(cudaMalloc(&d_S,  min(Nrows, Ncols) * sizeof(double)));

    // --- CUDA SVD initialization
    cusolveSafeCall(cusolverDnDgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size));
    double *work;   gpuErrchk(cudaMalloc(&work, work_size * sizeof(double)));

    printf("Solver Inicializado");

    cudaEvent_t start, stop ;
    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;

    //cudaEventRecord(start,0);

    clock_t launch = clock();

    // --- CUDA SVD execution
    cusolveSafeCall(cusolverDnDgesvd(solver_handle, 'A', 'A', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, d_V, Ncols, work, work_size, NULL, devInfo));
    int devInfo_h = 0;  gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) std::cout   << "Unsuccessful SVD execution\n\n";

    //cudaEventRecord(stop,0);
    //cudaEventSynchronize(stop);
    clock_t done = clock();
    float elapsedTime ;
    //cudaEventElapsedTime(&elapsedTime, start , stop) ;
    elapsedTime = launch - done ;
    printf("Operacion Terminada %f \n", elapsedTime);


    // --- Moving the results from device to host
    gpuErrchk(cudaMemcpy(h_S, d_S, min(Nrows, Ncols) * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_U, d_U, Nrows * Nrows     * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_V, d_V, Ncols * Ncols     * sizeof(double), cudaMemcpyDeviceToHost));


    /**
    std::cout << "Singular values\n";
    for(int i = 0; i < min(Nrows, Ncols); i++)
        std::cout << "d_S["<<i<<"] = " << std::setprecision(15) << h_S[i] << std::endl;

    std::cout << "\nLeft singular vectors - For y = A * x, the columns of U span the space of y\n";
    for(int j = 0; j < Nrows; j++) {
        printf("\n");
        for(int i = 0; i < Nrows; i++)
            printf("U[%i,%i]=%f\n",i,j,h_U[j*Nrows + i]);
    }

    std::cout << "\nRight singular vectors - For y = A * x, the columns of V span the space of x\n";
    for(int i = 0; i < Ncols; i++) {
        printf("\n");
        for(int j = 0; j < Ncols; j++)
            printf("V[%i,%i]=%f\n",i,j,h_V[j*Ncols + i]);
    }

    */

    cusolverDnDestroy(solver_handle);

   /******************************************/

	/********** VECTOR ADDITION

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


	*******************/

    printf("Done\n");
    return 0;
}

