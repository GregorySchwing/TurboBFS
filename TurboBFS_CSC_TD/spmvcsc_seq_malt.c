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
int spmv_seq_ug_csc_malt_a (float *f,int *I,int *CP,float *f_t,int n){

  int i, k, start, end;
  float sum;

  for (i=0; i<n; i++){
    f_t[i] = 0.0;
    sum = 0.0;
    start = CP[i];
    end = CP[i+1];
    printf("i %d start %d end %d \n", i, start, end);
    for (k=start; k<end; k++){
      printf("i %d start %d end %d k %d I[k] %d f[IK} %f \n", i, start, end, k, I[k], f[I[k]]);
      sum += f[I[k]];
    }
    if (sum > 0.1){
      printf("Expanding %d\n", i);
      f_t[i] = sum;
    }
  }

  return 0;
}//end spmv_seq_ug_csc


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
    if (m > -1){
      sum += f[m];
      printf("i %d m %d f[m] %f\n", i, m, f[m]);
    }
    if (sum > 0.1){
      f_t[i] = sum;
    }
  }

  return 0;
}//end spmv_seq_ug_csc
