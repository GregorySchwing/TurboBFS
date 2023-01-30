/*
 * This source code is distributed under the terms defined  
 * in the file bfstdcsc_main.c of this source distribution.
 */
/* 
 *  Breadth first search (BFS) 
 *  Single precision (float data type) 
 *  TurboBFS_CSC_TD:bfsgputdcsc_sc.cu
 * 
 *  This program computes the GPU-based parallel 
 *  top-down BFS (scalar) for unweighted graphs represented 
 *  by sparse adjacency matrices in the CSC format, including
 *  the computation of the S array to store the depth at 
 *  which each vertex is discovered.  
 *
 */

#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>

//includes CUDA project
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "matchgpu.h"
extern "C"{
                 #include "bfstdcsc.h"

}



/*************************prototype kernel*************************************/
__global__ void spMvUgCscScKernel (int *CP_d,int *IC_d,int *ft_d,int *f_d,
				   float *sigma_d,int j,int r,int n);
/******************************************************************************/

/* 
 * Function to compute a gpu-based parallel top-down BFS (scalar) for 
 * unweighted graphs represented by sparse adjacency matrices in CSC format,
 * including the computation of the S vector to store the depth at which each  
 * vertex is  discovered.
 *  
 */
int  bfs_gpu_mm_csc_sc (int *IC_h,int *CP_h,int *m_h,int nz,int n,int repetition){
  float t_spmv;
  float t_spmv_t = 0.0;
  float t_sum = 0.0;
  int i,dimGrid;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /*Allocate device memory for the vector CP_d */
  int *CP_d;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&CP_d),sizeof(*CP_d)*(n+1)));
  /*Copy host memory (CP_h) to device memory (CP_d)*/
  checkCudaErrors(cudaMemcpy(CP_d,CP_h,(n+1)*sizeof(*CP_d),cudaMemcpyHostToDevice));

  /*Allocate device memory for the vector IC_d */
  int *IC_d;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&IC_d),sizeof(*IC_d)*nz));
  /*Copy host memory (IC_h) to device memory (IC_d)*/
  checkCudaErrors(cudaMemcpy(IC_d,IC_h,nz*sizeof(*IC_d),cudaMemcpyHostToDevice));

  /*Allocate device memory for the vector m_d, and set m_d to zero. */
  int *m_d;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&m_d),sizeof(*m_d)*n));
  checkCudaErrors(cudaMemset(m_d,0,sizeof(*m_d)*n));

  /*Allocate device memory for the vector f_d*/
  int *req_d;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&req_d),sizeof(*req_d)*n));

  /*allocate unified memory for integer variable c for control of while loop*/
  int *c;
  checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&c),sizeof(*c)));

  /*computing MM */
  dimGrid = (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
  for (i = 0; i<repetition; i++){
    *c = 1;
    //d = 0;
    checkCudaErrors(cudaMemset(req_d,0,sizeof(*req_d)*n));
    checkCudaErrors(cudaMemset(m_d,0,sizeof(*m_d)*n));
    int count = 0;
    while (*c && ++count < NR_MAX_MATCH_ROUNDS){
      //d = d + 1;
      *c = 0;
      //printf("Iteration %d\n", count);
      cudaEventRecord(start);
      gaSelect<<<dimGrid,THREADS_PER_BLOCK>>>(m_d, c, n, rand());
      grRequest<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d,IC_d,req_d, m_d, n);
      grRespond<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d,IC_d,req_d, m_d, n);
      gMatch<<<dimGrid,THREADS_PER_BLOCK>>>(m_d, req_d, n);
      //spMvUgCscScKernel <<<dimGrid,THREADS_PER_BLOCK>>> (CP_d,IC_d,ft_d,f_d,sigma_d,d,r,n);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_spmv,start,stop);
      t_spmv_t += t_spmv;
      /*
      checkCudaErrors(cudaMemcpy(m_h,m_d,n*sizeof(*m_h),cudaMemcpyDeviceToHost));
      int matched = 0, red = 0, blue = 0, dead = 0;
      for (int i = 0; i < n; ++i){
        if (!m_h[i])
          ++blue;
        else if (m_h[i] == 1)
          ++red;
        else if (m_h[i] == 2)
          ++dead;
        else if (m_h[i] > 2)
          ++matched;
      }
      printf("it::red %d blue %d dead %d matched %d\n",red,blue,dead,matched);
      */
    }
  }
  printf("bfs_gpu_mm_csc_sc::t_sum=%lfms \n",t_spmv_t);

  /*Copy device memory (m_d) to host memory (S_h)*/
  checkCudaErrors(cudaMemcpy(m_h,m_d, n*sizeof(*m_d),cudaMemcpyDeviceToHost));

  int print_t = 1;
  if (print_t){
    printf("bfs_gpu_mm_csc_sc::average time mm d = %lfms \n",t_spmv_t/repetition);
  }

  /*cleanup memory*/
  checkCudaErrors(cudaFree(CP_d));
  checkCudaErrors(cudaFree(IC_d));
  checkCudaErrors(cudaFree(m_d));
  checkCudaErrors(cudaFree(req_d));
  checkCudaErrors(cudaFree(c));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return 0;
}//end bfs_gpu_td_csc_sc