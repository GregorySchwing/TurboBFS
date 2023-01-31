/* 
 * 
 * This program computes the sequential sparse matrix-vector 
 * multiplication for undirected, unweighted graphs represented 
 * by sparse adjacency matrices in the CSC format.
 *  
 */

#ifndef SPMVCSC_SEQ_H
#define SPMVCSC_SEQ_H

/* 
 * function to compute the sequential sparse matrix-vector multiplication for 
 * undirected, unweighted graphs represented by sparse adjacency matrices in
 * the CSC format.
 *   
 */
int spmv_seq_ug_csc (float *f,int *I,int *CP,float *f_t,int n);

/* 
 * function to compute the sequential sparse matrix-vector  multiplication for 
 * undirected, unweighted graphs represented by sparse adjacency matrices in
 * the CSC format.
 *   
 */
int spmv_seq_ug_csc_malt_a (float *f,int *I,int *CP,float *f_t,int n);
int spmv_seq_ug_csc_malt_b (float *f,int *I,int *CP,int *m_h,float *f_t,int n);

#endif
