
/*
 * Functions to:
 *   1) compute the parameters(maximum, mean, and standard
 *      deviation) of the degree, out degree and in degree
 *      vectors
 *   2) compute the degree vector for undirected graphs.
 *
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "degree.h"


/* 
 * function to compute the degree vector of an undirected graph represented by a 
 * symmetric sparse adjacency matrix.
 */
int degree_ug (int *out_degree,int *in_degree,int *degree,int N) {

  int i;

  for (i=0; i<N; i++){
    degree[i] = out_degree[i] + in_degree[i];
  }

  return 0;  
}// end degree_ug

/******************************************************************************/
/* 
 * function to compute the value and index of the maximum element of the degree
 * vector.
 *
*/   
int degree_max (int *degree,int *maxDegree,int *idxMaxDegree,int N){

  int i;
  for (i=0; i<N; i++){
      if (degree[i] >*maxDegree){
         *maxDegree = degree[i];
         *idxMaxDegree = i;
      }
    }
  
  return 0;
}// end degree_max

/******************************************************************************/
/*  function to compute: the mean value, the variance, and the standard deviation 
 *  of the values in the degree vector.  
 */    
int degree_sta (int *degree,double *meanDegree,double *sdDegree,int N){
  
  int i,count=0;
  int sum = 0;
  double sumsd = 0.0;
  double var; 
 
  /*computation of mean value*/
  for (i=0; i<N; i++){       
	sum += degree[i];
	 count++;      
  }
  if (count != 0){
    *meanDegree = (double)sum/((double)(count));
  }else{
    *meanDegree = 0;
  }

  /*computation of variance and standard deviation*/
  for (i=0; i<N; i++){
        sumsd += pow(((double)degree[i] - (*meanDegree)), 2);
    }
  var = sumsd/((double) (count-1));
  *sdDegree = sqrt (var);
  
  return 0;
}//end degree_sta

/******************************************************************************/
/* 
 * function to compute the degree distribution vector of a graph represented by a 
 * sparse adjacency matrix.
 */
int degree_dist (int *degree,int *degreeDist,int N) {

  int i;

  for (i=0; i<N; i++){
    degreeDist[degree[i]]++;
  }

  return 0;  
}// end degree_dist

/******************************************************************************/
/* 
 * function to compute the realtive degree distribution vector of a graph 
 * represented by a sparse adjacency matrix.
 */
int degree_dist_rel (int *degreeDist,double *degreeDistRel,int maxDegree,int N){
 int i;

  for (i=0; i<=maxDegree; i++){
    degreeDistRel[i] = (double) degreeDist[i]*100.0/((double) N) ;
  }

  return 0;  
}// end degree_dist_rel

/******************************************************************************/
/* 
 * function to compute the accumulated degree distribution and relative degree 
 * distribution vectors of a graph represented by a sparse adjacency matrix.
 * 
 */
int degree_dist_ac (int *degreeDist,double *degreeDistRel,int *degreeDistAc,
		    double *degreeDistRelAc,int maxDegree){

  
  int degree = 2,high = 4,low = 2; 
  for (int i=0; i<2; i++) {
    degreeDistAc[i] =degreeDist[i];
    degreeDistRelAc[i] =degreeDistRel[i];
  }
  
  while (low < maxDegree) {   
    while (degree <= maxDegree && (low <= degree && degree < high)) {      
      degreeDistAc[low] +=degreeDist[degree];
      degreeDistRelAc[low] +=degreeDistRel[degree];
      degree++;      
    }
    low =  2*low;
    high = 2*high;
  }
  return 0; 
}//end degree_dist_ac
