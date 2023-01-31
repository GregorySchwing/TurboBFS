/* 
*  Breadth first search (BFS) 
*  Single precision (float data type) 
* 
*  This program computes a sequential top-down
*  BFS for unweighted graphs represented 
*  by sparse matrices in the CSC format.
*
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include "spmvcsc_seq.h"
#include "bfstdcsc.h"

/* 
 * Function to compute a sequential top-down m-alternating BFS for unweighted graphs,
 * represented by sparse adjacency matrices in CSC format, including the  
 * computation of the S vector to store the depth at which each vertex is
 * discovered.    
 *  
*/

int bfs_seq_td_csc_malt (int *IC,int *CP,int *S,float *sigma,int *m_h,int nz,int n){

    int d = 0;
    int c = 1;
    float *f;
    float *f_t;
    f =  (float *) calloc(n,sizeof(*f));
    f_t =  (float *) calloc(n,sizeof(*f_t));
    // Set multiple srcs.
    for (int r = 0; r < n; ++r){
      S[r] = 0;
      sigma[r] = 0.0;
      if (m_h[r] < 0){
        f[r] = 1;
      }
    }
    for (int r = 0; r < n; ++r){
        printf("%f ",f[r]);
    }
    //f[r] = 1;
    /* timing variables  */
    double initial_t;
    double delta;
    double sum_vs_t = 0;
    double spmv_t = 0;
    double assign_v_t = 0;
    double mult_vs_t = 0;
    double S_v_t = 0;
    double check_f_t = 0;
    double total_t;
    
    while (c > 0) {
      d++;
      initial_t = get_time();
      sum_vs (sigma,f,n);
      delta = get_time()-initial_t;
      sum_vs_t += delta;
      
      initial_t = get_time();
      if ((d-1)%2 == 0){
        printf("calling spmv_seq_ug_csc d %d\n", d);
        spmv_seq_ug_csc_malt_a (f,IC,CP,f_t,n);
      } else {
        printf("calling spmv_seq_ug_csc_malt_b d %d\n", d);
        spmv_seq_ug_csc_malt_b (f,IC,CP,m_h,f_t,n);
      }
      delta = get_time()-initial_t;
      spmv_t += delta;
      for (int r = 0; r < n; ++r){
        printf("%f ",f[r]);
      }
      printf("\n"); 
      initial_t = get_time();
      assign_v(f,f_t,n);
      delta = get_time()-initial_t;
      assign_v_t += delta;
      for (int r = 0; r < n; ++r){
        printf("%f ",f[r]);
      }
      printf("\n"); 
      initial_t = get_time();
      mult_vs (sigma,f,n);
      delta = get_time()-initial_t;
      mult_vs_t += delta;
printf("Multi\n");
      for (int r = 0; r < n; ++r){
        printf("%f ",f[r]);
      }
      printf("\n");
      initial_t = get_time();
      S_v (S,f,n,d);
      delta = get_time()-initial_t;
      S_v_t += delta;
      for (int r = 0; r < n; ++r){
        printf("%f ",f[r]);
      }
      printf("\n");
      initial_t = get_time();
      c = 0;
      check_f(&c,f,n);
      delta = get_time()-initial_t;
      check_f_t += delta;
    }

    printf("\nbfs_seq_ug_csc_malt::d = %d,\n",d);
    total_t =  sum_vs_t +  spmv_t + assign_v_t +  mult_vs_t + S_v_t + check_f_t; 
    int p_t = 1;
    if (p_t) {    
      printf("bfstdcsc_seq_malt::f <-- f +sigma time = %lfs \n",sum_vs_t);
      printf("bfstdcsc_seq_malt::f_t <-- fA time = %lfs \n",spmv_t);
      printf("bfstdcsc_seq_malt::f <-- f_t time = %lfs \n",assign_v_t);
      printf("bfstdcsc_seq_malt::f <-- f*(-sigma) time = %lfs \n",mult_vs_t);
      printf("bfstdcsc_seq_malt::S vector time = %lfs \n",S_v_t);
      printf("bfstdcsc_seq_malt::c <-- check (f=0)) time = %lfs \n",check_f_t);
      printf("bfstdcsc_seq_malt::sequential top-down BFS total time = %lfs \n",total_t);
    }
    
    return 0;
}//end  bfs_seq_td_csc
