/* 
 *  Breadth first search (BFS) 
 *  Single precision (float data type) 
 * 
 * This program computes the sequential sparse matrix-vector 
 * multiplication for undirected, unweighted graphs represented 
 * by sparse adjacency matrices in the CSC format.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "spmvcsc_seq.h"

/* 
 * function to compute the sequential sparse matrix-vector multiplication for 
 * undirected, unweighted graphs represented by sparse adjacency matrices in
 * the CSC format.
 *   
 */

int spmv_seq_ug_csc_malt_b (float *f,int *I,int *CP,int *m_h,float *f_t,int n){

  int i;
  float sum;

  for (i=0; i<n; i++){
    f_t[i] = 0.0;
    sum = 0.0;
    int m = m_h[i];
    if (m > -1)
      sum += f[I[m]];
    if (sum > 0.1){
      f_t[i] = sum;
    }
  }

  return 0;
}//end spmv_seq_ug_csc
