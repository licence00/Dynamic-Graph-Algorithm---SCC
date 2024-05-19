#ifndef SCC_H
#define SCC_H

#include <iostream>
#include <climits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>


__global__ void scc_fw_kernel0(int V, int E, int* dOffsetArray, int* dEdgelist, int* dSrcList, int *dRevOffsetArray, bool* dVisit, int* dOutDeg, int* dInDeg,int*scc,int*scc_offset
            ,int*sccCount,int*sccindex,int*sccoffset,bool*noNewNode)
{ 
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (dVisit[v] == false) {
    int outNbrCount = 0;
    int inNbrCount = 0;
    for (int edge = dOffsetArray[v]; edge < dOffsetArray[v+1]; edge++) { // FOR NBR ITR 
      int dst = dEdgelist[edge];
      if (dVisit[dst] == false) {
        outNbrCount = outNbrCount + 1;

      } 

    } 
    for (int edge = dRevOffsetArray[v]; edge < dRevOffsetArray[v+1]; edge++)
    {
      int dst = dSrcList[edge] ;
      if (dVisit[dst] == false) {
        inNbrCount = inNbrCount + 1;

      } 

    } 
    if (inNbrCount == 0 || outNbrCount == 0) {
      dVisit[v] = true;
      atomicAdd(sccCount,1);
      scc[atomicAdd(sccindex,1)]= v;
      scc_offset[atomicAdd(sccoffset,1)]= 1;
      *noNewNode = false;
    } 
    dInDeg[v] = inNbrCount;
    dOutDeg[v] = outNbrCount;

  } 
} 

__global__ void scc_fw_kernel1(int V, int E, bool* dVisit, int* dInDeg, int* dOutDeg,int*maxDegree)
{ 
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (dVisit[v] == false) {
    int neighbourCount = dInDeg[v] + dOutDeg[v];
    if (*maxDegree < neighbourCount) {
      *maxDegree = neighbourCount;
    } 

  } 
} 

__global__ void scc_fw_kernel2(int V, int E, bool* dVisit, int* dInDeg, int* dOutDeg,int*maxDegree,int*nodeToBeVisited)
{ 
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (dVisit[v] == false) {
    int curNodeDegree = dInDeg[v] + dOutDeg[v];
    if (curNodeDegree == *maxDegree) {
      *nodeToBeVisited = v;
    } 

  } 
} 

__global__ void scc_fw_kernel3(int V, int E, int* dVisitLevelBw, int* dVisitLevelFw,int*nodeToBeVisited){ 
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (v == *nodeToBeVisited) {
    dVisitLevelFw[v] = 1;
    dVisitLevelBw[v] = 1;
  } 
} 

__global__ void fwd_pass(int n, int* dOffsetArray, int* dEdgelist, bool* dFinished, int* dVisitLevelFw, bool* dVisit, bool* dVisitFw) {
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= n) return;
  if (dVisitLevelFw[v] == 1 && dVisit[v] == false) {
    dVisitFw[v] = true;
    for (int edge = dOffsetArray[v]; edge < dOffsetArray[v+1]; edge++) { 
      int w = dEdgelist[edge];
      if (dVisitFw[w] == false && dVisitLevelFw[w] == 0) {
        dVisitLevelFw[w] = 1;
        *dFinished = false;
      } 
    } 
    dVisitLevelFw[v] = 2;

  } 
} 

__global__ void fwd_pass2(int n, int* dRevOffsetArray, int* dSrclist, bool* dFinished2, int* dVisitLevelBw, bool* dVisit, bool* dVisitFw, bool* dVisitBw) {
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= n) return;
  if (dVisitLevelBw[v] == 1 && dVisit[v] == false && dVisitFw[v] == true) {
    dVisitBw[v] = true; 
    for (int edge = dRevOffsetArray[v]; edge < dRevOffsetArray[v+1]; edge++) { 
      int w = dSrclist[edge];
      if (dVisitBw[w] == false && dVisitLevelBw[w] == 0) {
        dVisitLevelBw[w] = 1;
        *dFinished2 = false;
      } 
    } 
    dVisitLevelBw[v] = 2;

  } 
} 

__global__ void scc_fw_kernel4(int V, int E, bool* dVisit, bool* dVisitFw, bool* dVisitBw,int*sccsize,int*scc,int*scc_offset,bool*noNewComp,int*sccindex){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (dVisit[v] == false && dVisitFw[v] && dVisitBw[v]) 
  {
    dVisit[v] = true;
    *noNewComp = false;
    atomicAdd(sccsize,1);
    scc[atomicAdd(sccindex,1)] = v;
  }  
  else 
  {
    dVisitFw[v] = false;
    dVisitBw[v] = false;
  }

} 

template <typename T>
__global__ void initKernel(unsigned V, T* init_array, T init_value) 
{  
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < V) {
    init_array[id] = init_value;
  }
}

#endif
