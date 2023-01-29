/*
 * This source code is distributed under the terms defined  
 * in the file bfstdcsc_main.c of this source distribution.
*/
/* 
 *  Breadth first search (BFS) 
 *  Single precision (float data type) 
 *  TurboBFS_CSC_TD:bfsgputdcsc_wa.cu
 * 
 *  This program computes the GPU-based parallel top-down
 *  BFS (warp shuffle) for unweighted graphs represented by 
 *  sparse adjacency matrices in the CSC format, including
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

/*************************prototype kernel MVS*********************************/
void spMvUgCscWaKernel (int *CP_d,int *IC_d,int *ft_d,int *f_d,
				   float *sigma_d,int d,int r,int n, sycl::nd_item<3> item_ct1,
				   sycl::local_accessor<volatile int, 2> cp);
/******************************************************************************/

/* 
 * function to compute a gpu-based parallel top-down BFS (warp shuffle) for
 * unweighted graphs represented by sparse adjacency matrices in CSC format,
 * including the computation of the S vector to store the depth at which each 
 * vertex is discovered.
 *  
 */
int bfs_gpu_td_csc_wa(int *IC_h, int *CP_h, int *S_h, float *sigma_h, int r,
                      int nz, int n, int repetition) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  float t_spmv;
  float t_spmv_t = 0.0;
  float t_bfsfunctions;
  float t_bfsfunctions_t = 0.0;
  float t_sum = 0.0;
  float t_bfs_avg;
  int i, d,  dimGrid_mvsp, dimGrid;
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  start = new sycl::event();
  stop = new sycl::event();

  /*Allocate device memory for the vector CP_d */
  int *CP_d;
  /*
  DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (CP_d = (int *)sycl::malloc_device(sizeof(*CP_d) * (n + 1), q_ct1), 0));
  /*Copy host memory (CP_h) to device memory (CP_d)*/
  /*
  DPCT1003:33: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (q_ct1.memcpy(CP_d, CP_h, (n + 1) * sizeof(*CP_d)).wait(), 0));

  /*Allocate device memory for the vector IC_d */
  int *IC_d;
  /*
  DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (IC_d = (int *)sycl::malloc_device(sizeof(*IC_d) * nz, q_ct1), 0));
  /*Copy host memory (IC_h) to device memory (IC_d)*/
  /*
  DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((q_ct1.memcpy(IC_d, IC_h, nz * sizeof(*IC_d)).wait(), 0));

  /*Allocate device memory for the vector S_d, and set S_d to zero. */
  int *S_d;
  /*
  DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (S_d = (int *)sycl::malloc_device(sizeof(*S_d) * n, q_ct1), 0));
  ((q_ct1.memset(S_d, 0, sizeof(*S_d) * n).wait(), 0));

  /*Allocate device memory for the vector sigma_d */
  float *sigma_d;
  /*
  DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (sigma_d = (float *)sycl::malloc_device(sizeof(*sigma_d) * n, q_ct1), 0));

  /*Allocate device memory for the vector f_d*/
  int *f_d;
  /*
  DPCT1003:38: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (f_d = (int *)sycl::malloc_device(sizeof(*f_d) * n, q_ct1), 0));

  /*Allocate device memory for the vector ft_d*/
  int *ft_d;
  /*
  DPCT1003:39: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (ft_d = (int *)sycl::malloc_device(sizeof(*ft_d) * n, q_ct1), 0));

  /*allocate unified memory for integer variable c for control of while loop*/
  int *c;
  /*
  DPCT1003:40: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((c = (int *)sycl::malloc_shared(sizeof(*c), q_ct1), 0));

  /*computing BFS */
  dimGrid = (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
  dimGrid_mvsp = (n + THREADS_PER_WARP)/THREADS_PER_WARP;
  for (i = 0; i<repetition; i++){
    *c = 1;
    d = 0;
    ((q_ct1.memset(f_d, 0, sizeof(*f_d) * n).wait(), 0));
    ((q_ct1.memset(sigma_d, 0, sizeof(*sigma_d) * n).wait(), 0));
    while (*c){
      d = d+1;
      *c = 0;

      /*
      DPCT1012:41: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      start_ct1 = std::chrono::steady_clock::now();
      *start = q_ct1.ext_oneapi_submit_barrier();
      /*
      DPCT1003:45: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      ((q_ct1.memset(ft_d, 0, sizeof(*ft_d) * n).wait(), 0));
      /*
      DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<volatile int, 2> cp_acc_ct1(
            sycl::range<2>(32 /*(THREADS_PER_BLOCK/THREADS_PER_WARP)*/, 2),
            cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, dimGrid_mvsp) *
                                  sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                              sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              spMvUgCscWaKernel(CP_d, IC_d, ft_d, f_d, sigma_d, d, r, n,
                                item_ct1, cp_acc_ct1);
            });
      });
      /*
      DPCT1012:42: Detected kernel execution time measurement pattern and
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
      DPCT1012:43: Detected kernel execution time measurement pattern and
      generated an initial code for time measurements in SYCL. You can change
      the way time is measured depending on your goals.
      */
      start_ct1 = std::chrono::steady_clock::now();
      *start = q_ct1.ext_oneapi_submit_barrier();
      /*
      DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
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
      DPCT1012:44: Detected kernel execution time measurement pattern and
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
  printf("\nbfsgputdcsc_wa::d = %d,r = %d,t_sum=%lfms \n",d,r,t_sum);
  t_bfs_avg = t_sum/repetition;

  /*Copy device memory (sigma_d) to host memory (sigma_h)*/
  /*
  DPCT1003:46: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  (
      (q_ct1.memcpy(sigma_h, sigma_d, n * sizeof(*sigma_d)).wait(), 0));

  /*Copy device memory (S_d) to host memory (S_h)*/
  /*
  DPCT1003:47: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((q_ct1.memcpy(S_h, S_d, n * sizeof(*S_d)).wait(), 0));

  int print_t = 1;
  if (print_t){
    printf("bfsgputdcsc_wa::time f <-- fA = %lfms \n",t_spmv_t/repetition);
    printf("bfsgputdcsc_wa::time time bfs functions = %lfms \n", t_bfsfunctions_t/repetition);
    printf("bfsgputdcsc_wa::average time BFS  = %lfms \n",t_bfs_avg);
  }

  /*cleanup memory*/
  ((sycl::free(CP_d, q_ct1), 0));
  ((sycl::free(IC_d, q_ct1), 0));
  ((sycl::free(S_d, q_ct1), 0));
  ((sycl::free(sigma_d, q_ct1), 0));
  ((sycl::free(f_d, q_ct1), 0));
  ((sycl::free(ft_d, q_ct1), 0));
  ((sycl::free(c, q_ct1), 0));
  ((dpct::destroy_event(start), 0));
  /*
  DPCT1003:48: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  ((dpct::destroy_event(stop), 0));

  return 0;
}//end bfs_gpu_td_csc_wa

/**************************************************************************/
/* 
 * if d = 1, initialize f(r) and sigma(r),
 * compute the gpu-based parallelsparse matrix-vector multiplication    
 * for sparse matrices in the CSC format, representing unweighted 
 * graphs. 
 */

void spMvUgCscWaKernel (int *CP_d,int *IC_d,int *ft_d,int *f_d,
			float *sigma_d,int d,int r,int n, sycl::nd_item<3> item_ct1,
			sycl::local_accessor<volatile int, 2> cp){

  //initialize f(r) and sigma(r)
  if (d == 1){
      f_d[r] = 1;
      sigma_d[r] = 1.0;
  }
  //compute spmv
  int thread_id = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) *
                      item_ct1.get_local_range(2);       // global thread index
  int thread_lane_id = thread_id & (THREADS_PER_WARP-1); //thread index within the warp
  int warp_id = thread_id/THREADS_PER_WARP; //global warp index
  int warp_lane_id = item_ct1.get_local_id(2) /
                     THREADS_PER_WARP; // warp index within the block
  int num_warps = WARPS_PER_BLOCK * item_ct1.get_group_range(
                                        2); // total number of available warps

  int col;
  int icp;
  unsigned mask = 0xffffffff;

  for (col = warp_id; col < n; col += num_warps){
    if(sigma_d[col] < 0.01) {
    
      if (thread_lane_id<2){
        cp[warp_lane_id][thread_lane_id] = CP_d[col+thread_lane_id];
      }
      int start = cp[warp_lane_id][0];
      int end = cp[warp_lane_id][1];
      //printf("threadIdx.x=%d,blockIdx.x=%d,thread_lane_id=%d,warp_id=%d,warp_lane_id=%d,num_warps=%d,col=%d,start=%d,end=%d\n",threadIdx.x,blockIdx.x,thread_lane_id,warp_id,warp_lane_id,num_warps,col,start,end);
      int sum = 0;

      if (end - start > THREADS_PER_WARP){//number of column elements > THREADS_PER_WARP
        icp = start -(start & (THREADS_PER_WARP-1)) + thread_lane_id;
        /*accumulate local sums*/
        if (icp >= start && icp < end){
	  sum += f_d[IC_d[icp]];
        }
        /*accumulate local sums*/
        for (icp += THREADS_PER_WARP; icp < end; icp += THREADS_PER_WARP){
          sum += f_d[IC_d[icp]];
        }
      }else{//number of column elements <= THREADS_PER_WARP
        /*accumulate local sums*/
        for (icp = start + thread_lane_id; icp < end; icp += THREADS_PER_WARP){
          sum += f_d[IC_d[icp]];
        }
      }
      /*reduce local sums by the warp shuffle instruction */
      for (int offset = THREADS_PER_WARP/2; offset > 0; offset /= 2){
        /*
        DPCT1023:4: The SYCL sub-group does not support mask options for
        dpct::shift_sub_group_left.
        */
        sum +=
            dpct::shift_sub_group_left(item_ct1.get_sub_group(), sum, offset);
      }
      /*first thread in the warp output the final result*/
      if (thread_lane_id == 0){
        ft_d[col] = sum;
      }   
    }
  }
}//end spMvUgCscWaKernel
