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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>

//includes CUDA project
#include "bfsgputdcsc.dp.hpp"

#include <chrono>
#include "bfstdcsc.h"



/*************************prototype kernel*************************************/
void spMvUgCscScKernel (int *CP_d,int *IC_d,int *ft_d,int *f_d,
				   float *sigma_d,int j,int r,int n, sycl::nd_item<3> item_ct1);
/******************************************************************************/

/* 
 * Function to compute a gpu-based parallel top-down BFS (scalar) for 
 * unweighted graphs represented by sparse adjacency matrices in CSC format,
 * including the computation of the S vector to store the depth at which each  
 * vertex is  discovered.
 *  
 */
int bfs_gpu_td_csc_sc(int *IC_h, int *CP_h, int *S_h, float *sigma_h, int r,
                      int nz, int n, int repetition) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  float t_spmv;
  float t_spmv_t = 0.0;
  float t_bfsfunctions;
  float t_bfsfunctions_t = 0.0;
  float t_sum = 0.0;
  float t_bfs_avg;
  int i,d = 0,dimGrid;
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  start = new sycl::event();
  stop = new sycl::event();

  /*Allocate device memory for the vector CP_d */
  int *CP_d;
  /*
  DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (CP_d = (int *)sycl::malloc_device(sizeof(*CP_d) * (n + 1), q_ct1), 0));
  /*Copy host memory (CP_h) to device memory (CP_d)*/
  /*
  DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (q_ct1.memcpy(CP_d, CP_h, (n + 1) * sizeof(*CP_d)).wait(), 0));

  /*Allocate device memory for the vector IC_d */
  int *IC_d;
  /*
  DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (IC_d = (int *)sycl::malloc_device(sizeof(*IC_d) * nz, q_ct1), 0));
  /*Copy host memory (IC_h) to device memory (IC_d)*/
  /*
  DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((q_ct1.memcpy(IC_d, IC_h, nz * sizeof(*IC_d)).wait(), 0));

  /*Allocate device memory for the vector S_d, and set S_d to zero. */
  int *S_d;
  /*
  DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (S_d = (int *)sycl::malloc_device(sizeof(*S_d) * n, q_ct1), 0));
  /*
  DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((q_ct1.memset(S_d, 0, sizeof(*S_d) * n).wait(), 0));

  /*Allocate device memory for the vector sigma_d */
  float *sigma_d;
  /*
  DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (sigma_d = (float *)sycl::malloc_device(sizeof(*sigma_d) * n, q_ct1), 0));

  /*Allocate device memory for the vector f_d*/
  int *f_d;
  /*
  DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (f_d = (int *)sycl::malloc_device(sizeof(*f_d) * n, q_ct1), 0));

  /*Allocate device memory for the vector ft_d*/
  int *ft_d;
  /*
  DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (ft_d = (int *)sycl::malloc_device(sizeof(*ft_d) * n, q_ct1), 0));

  /*allocate unified memory for integer variable c for control of while loop*/
  int *c;
  /*
  DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((c = (int *)sycl::malloc_shared(sizeof(*c), q_ct1), 0));

  /*computing BFS */
  dimGrid = (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
  for (i = 0; i<repetition; i++){
    *c = 1;
    d = 0;
    /*
    DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    ((q_ct1.memset(f_d, 0, sizeof(*f_d) * n).wait(), 0));
    /*
    DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    ((q_ct1.memset(sigma_d, 0, sizeof(*sigma_d) * n).wait(), 0));
    while (*c){
      d = d + 1;
      *c = 0;

      /*
      DPCT1012:17: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      start_ct1 = std::chrono::steady_clock::now();
      *start = q_ct1.ext_oneapi_submit_barrier();
      /*
      DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      q_ct1.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, dimGrid) *
                                sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
          [=](sycl::nd_item<3> item_ct1) {
            spMvUgCscScKernel(CP_d, IC_d, ft_d, f_d, sigma_d, d, r, n,
                              item_ct1);
          });
      /*
      DPCT1012:18: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      dpct::get_current_device().queues_wait_and_throw();
      stop_ct1 = std::chrono::steady_clock::now();
      *stop = q_ct1.ext_oneapi_submit_barrier();
      t_spmv = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                   .count();
      t_spmv_t += t_spmv;

      /*
      DPCT1012:19: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      start_ct1 = std::chrono::steady_clock::now();
      *start = q_ct1.ext_oneapi_submit_barrier();
      /*
      DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      q_ct1.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, dimGrid) *
                                sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
          [=](sycl::nd_item<3> item_ct1) {
            bfsFunctionsKernel(f_d, ft_d, sigma_d, S_d, c, n, d, item_ct1);
          });
      /*
      DPCT1012:20: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      dpct::get_current_device().queues_wait_and_throw();
      stop_ct1 = std::chrono::steady_clock::now();
      *stop = q_ct1.ext_oneapi_submit_barrier();
      t_bfsfunctions =
          std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
              .count();
      t_bfsfunctions_t += t_bfsfunctions;
      
      t_sum += t_spmv + t_bfsfunctions;
    }
  }
  printf("\nbfsgputdcsc_sc::d = %d,r = %d,t_sum=%lfms \n",d,r,t_sum);
  t_bfs_avg = t_sum/repetition;

  /*Copy device memory (sigma_d) to host memory (sigma_h)*/
  /*
  DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (q_ct1.memcpy(sigma_h, sigma_d, n * sizeof(*sigma_d)).wait(), 0));

  /*Copy device memory (S_d) to host memory (S_h)*/
  /*
  DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((q_ct1.memcpy(S_h, S_d, n * sizeof(*S_d)).wait(), 0));

  int print_t = 1;
  if (print_t){
    printf("bfsgputdcsc_sc::time f <-- fA d = %lfms \n",t_spmv_t/repetition);
    printf("bfsgputdcsc_sc::time time bfs functions d = %lfms \n", t_bfsfunctions_t/repetition);
    printf("bfsgputdcsc_sc::average time BFS d = %lfms \n",t_bfs_avg);
  }

  /*cleanup memory*/
  /*
  DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((sycl::free(CP_d, q_ct1), 0));
  /*
  DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((sycl::free(IC_d, q_ct1), 0));
  /*
  DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((sycl::free(S_d, q_ct1), 0));
  /*
  DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((sycl::free(sigma_d, q_ct1), 0));
  /*
  DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((sycl::free(f_d, q_ct1), 0));
  /*
  DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((sycl::free(ft_d, q_ct1), 0));
  /*
  DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((sycl::free(c, q_ct1), 0));
  /*
  DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((dpct::destroy_event(start), 0));
  /*
  DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((dpct::destroy_event(stop), 0));

  return 0;
}//end bfs_gpu_td_csc_sc


/**************************************************************************/
/* 
 * if d = 1, initialize f(r) and sigma(r),
 * compute the gpu-based parallel sparse matrix-vector multiplication    
 * for sparse matrices in the CSC format, representing unweighted 
 * graphs. 
 */

void spMvUgCscScKernel (int *CP_d,int *IC_d,int *ft_d,int *f_d,
			float *sigma_d,int d,int r,int n, sycl::nd_item<3> item_ct1){

  int i = item_ct1.get_local_id(2) +
          item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if(i < n){
    //initialize f(r) and sigma(r)
    if (d == 1){
      f_d[r] = 1;
      sigma_d[r] = 1.0;
    }
    //compute spmv
    ft_d[i] = 0;
    if (sigma_d[i] < 0.01){
      int k;
      int start = CP_d[i];
      int end = CP_d[i+1];
      int sum = 0;
      for (k = start;k < end; k++){
	sum += f_d[IC_d[k]];
      }
      if (sum > 0.9) {
	ft_d[i] = sum;
      }
    }
  }
}//end spMvUgCscScKernel

/**************************************************************************/
/*
 * assign vector ft_d to vector f_d,
 * check that the vector f_d  has at least one nonzero element
 * add the vector f to vector sigma.
 * compute the S vector. 
 */
SYCL_EXTERNAL
void bfsFunctionsKernel (int *f_d,int *ft_d,float *sigma_d,int *S_d,int *c,
			 int n,int d, sycl::nd_item<3> item_ct1){

  int i = item_ct1.get_local_id(2) +
          item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (i < n){
    f_d[i] = 0;
    if (ft_d[i] > 0.9){
      *c = 1;
      f_d[i] = ft_d[i];
      sigma_d[i] += ft_d[i];
      S_d[i] = d;
    }
  }
}//end  bfsFunctionsKernel
