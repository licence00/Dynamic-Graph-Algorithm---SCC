#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "SCC.h"
using namespace std;

__global__ void fill_the_rs(int2* d_vec,int* d_fc,int* d_fr,int* d_bc,int* d_br,int E)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<E)
  {
    atomicAdd(&d_fr[d_vec[i].x],1);
    atomicAdd(&d_br[d_vec[i].y],1);
  }
}

__global__ void fill_the_cs(int2* d_vec,int* d_fc,int* d_fr,int* d_bc,int* d_br,int*temp_f,int*temp_b,int E)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<E)
  {
    int from = d_vec[i].x;
    int to = d_vec[i].y;
    d_fc[d_fr[from] + atomicAdd(&temp_f[from],1)] = to;
    d_bc[d_br[to] + atomicAdd(&temp_b[to],1)] = from;
  }
}

class csr_format
{
  public:
  int Edges,Vertices,CSize,RSize;
  int*d_fc_ptr=nullptr; int*d_fr_ptr=nullptr; int*d_bc_ptr=nullptr; int*d_br_ptr=nullptr;
  thrust::device_vector<int>Fr,Fc,Br,Bc;
  csr_format(int vertices,int edges)
  {
    this->Vertices = vertices;
    this->Edges = edges;
    this->RSize = vertices+1;
    this->CSize = edges;
    Fr = thrust::device_vector<int>(RSize,0);
    Fc = thrust::device_vector<int>(CSize,0);
    Br = thrust::device_vector<int>(RSize,0);
    Bc = thrust::device_vector<int>(CSize,0);
    d_fc_ptr = thrust::raw_pointer_cast(Fc.data());
    d_fr_ptr = thrust::raw_pointer_cast(Fr.data());
    d_bc_ptr = thrust::raw_pointer_cast(Bc.data());
    d_br_ptr = thrust::raw_pointer_cast(Br.data());
  }
  bool mycomp(pair<int, int> a, pair<int , int> b)
  {
    return (a.first<b.first || (a.first==b.first && a.second<b.second));
  } 
  void create_csr_format(vector<pair<int,int>>&edgeList)
  {
    thrust::device_vector<int2>d_vec(edgeList.size());
    for(int i=0;i<edgeList.size();i++)
    {
      d_vec[i] = make_int2(edgeList[i].first,edgeList[i].second);
    }

    int2* d_vec_ptr = thrust::raw_pointer_cast(d_vec.data());

    int blocksize = 512;
    int blocks = (this->Edges + blocksize - 1) / blocksize;
    fill_the_rs<<<blocks,blocksize>>>(d_vec_ptr,d_fc_ptr,d_fr_ptr,d_bc_ptr,d_br_ptr,this->Edges); 
    cudaDeviceSynchronize();

    thrust::exclusive_scan(Fr.begin(),Fr.end(),Fr.begin());
    thrust::exclusive_scan(Br.begin(),Br.end(),Br.begin());

    thrust::device_vector<int>temp_f(this->Vertices,0);
    thrust::device_vector<int>temp_b(this->Vertices,0);

    int*temp_f_ptr = thrust::raw_pointer_cast(temp_f.data());
    int*temp_b_ptr = thrust::raw_pointer_cast(temp_b.data());

    fill_the_cs<<<blocks,blocksize>>>(d_vec_ptr,d_fc_ptr,d_fr_ptr,d_bc_ptr,d_br_ptr,temp_f_ptr,temp_b_ptr,this->Edges);
    cudaDeviceSynchronize();
  }
  void print()
  {
      int* tempFr = new int[RSize];
      int* tempFc = new int[CSize];
      int* tempBc = new int[CSize];
      int* tempBr = new int[RSize];

      // Copy GPU memory back to CPU
      cudaMemcpy(tempFr, d_fr_ptr, RSize * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(tempFc, d_fc_ptr, CSize * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(tempBc, d_bc_ptr, CSize * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(tempBr, d_br_ptr, RSize * sizeof(int), cudaMemcpyDeviceToHost);

      // Now you can print the data safely
      for(int i = 0; i < CSize; i++)
      {
          printf("%d ", tempFc[i]);
      }
      printf("\n");
      for(int i = 0; i < RSize; i++)
      {
          printf("%d ", tempFr[i]);
      }
      printf("\n");
      for(int i = 0; i < CSize; i++)
      {
          printf("%d ", tempBc[i]);
      }
      printf("\n");
      for(int i = 0; i < RSize; i++)
      {
          printf("%d ", tempBr[i]);
      }
      printf("\n");

      // Don't forget to free the temporary host memory
      delete[] tempFr;
      delete[] tempFc;
      delete[] tempBc;
      delete[] tempBr;
  }
  void free_csr_format()
  {
    Fr.clear(); Fc.clear(); Br.clear(); Bc.clear();
  }
};

void SCC(csr_format &csr,int*scc,int*scc_offset)
{
  int V = csr.Vertices;
  int E = csr.Edges;
  const unsigned threadsPerBlock = 512;
  const unsigned numThreads = (V < threadsPerBlock) ? V : 512;
  const unsigned numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;

  int *dInDeg; cudaMalloc(&dInDeg, sizeof(int) * (V));
  int *dOutDeg; cudaMalloc(&dOutDeg, sizeof(int) * (V));
  bool *dVisit; cudaMalloc(&dVisit, sizeof(bool) * (V));
  bool *dVisitFw; cudaMalloc(&dVisitFw, sizeof(bool) * (V));
  bool *dVisitBw; cudaMalloc(&dVisitBw, sizeof(bool) * (V));
  int *dVisitLevelFw; cudaMalloc(&dVisitLevelFw, sizeof(int) * (V));
  int *dVisitLevelBw; cudaMalloc(&dVisitLevelBw, sizeof(int) * (V));
  initKernel<int><<<numBlocks, numThreads>>>(V, dInDeg, 0);
  initKernel<int><<<numBlocks, numThreads>>>(V, dOutDeg, 0);
  initKernel<bool><<<numBlocks, numThreads>>>(V, dVisit, false);
  initKernel<bool><<<numBlocks, numThreads>>>(V, dVisitFw, false);
  initKernel<bool><<<numBlocks, numThreads>>>(V, dVisitBw, false);
  bool noNewComp = false;
  int sccCount = 0;
  int sccindex = 0;
  int sccoffset = 0;

  bool*noNewComp_d; cudaMalloc(&noNewComp_d,sizeof(bool));
  bool*noNewNode_d; cudaMalloc(&noNewNode_d,sizeof(bool));
  int*sccCount_d; cudaMalloc(&sccCount_d,sizeof(int));
  int*sccindex_d; cudaMalloc(&sccindex_d,sizeof(int));
  int*sccoffset_d; cudaMalloc(&sccoffset_d,sizeof(int));
  int*nodeToBeVisited_d; cudaMalloc(&nodeToBeVisited_d,sizeof(int));
  int*maxDegree_d; cudaMalloc(&maxDegree_d,sizeof(int));

  cudaMemcpy(sccindex_d, &sccindex, sizeof(int), cudaMemcpyHostToDevice);

  while(!noNewComp) {
    noNewComp = true;
    cudaMemcpy(noNewComp_d, &noNewComp, sizeof(bool), cudaMemcpyHostToDevice);
    bool noNewNode = false;

    while(!noNewNode) 
    {
      noNewNode = true;
      cudaMemcpy(noNewNode_d, &noNewNode, sizeof(bool), cudaMemcpyHostToDevice);
      cudaMemcpy(sccCount_d, &sccCount, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(sccoffset_d, &sccoffset, sizeof(int), cudaMemcpyHostToDevice);
      scc_fw_kernel0<<<numBlocks, threadsPerBlock>>>(V, E, csr.d_fr_ptr, csr.d_fc_ptr, csr.d_bc_ptr,csr.d_br_ptr, dVisit, dOutDeg, dInDeg,scc,scc_offset,sccCount_d,sccindex_d,sccoffset_d,noNewNode_d);
      cudaDeviceSynchronize();
      cudaMemcpy(&noNewNode, noNewNode_d, sizeof(bool), cudaMemcpyDeviceToHost);
      cudaMemcpy(&sccCount, sccCount_d, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&sccoffset, sccoffset_d, sizeof(int), cudaMemcpyDeviceToHost);
    }

    int nodeToBeVisited = -1;
    int maxDegree = -1;
    cudaMemcpy(maxDegree_d, &maxDegree, sizeof(int), cudaMemcpyHostToDevice);

    scc_fw_kernel1<<<numBlocks, threadsPerBlock>>>(V, E, dVisit, dInDeg, dOutDeg,maxDegree_d);
    cudaDeviceSynchronize();

    cudaMemcpy(nodeToBeVisited_d, &nodeToBeVisited, sizeof(int), cudaMemcpyHostToDevice);   

    scc_fw_kernel2<<<numBlocks, threadsPerBlock>>>(V, E, dVisit, dInDeg, dOutDeg,maxDegree_d,nodeToBeVisited_d);
    cudaDeviceSynchronize();

    initKernel<int><<<numBlocks, numThreads>>>(V, dVisitLevelFw, 0);
    initKernel<int><<<numBlocks, numThreads>>>(V, dVisitLevelBw, 0);

    scc_fw_kernel3<<<numBlocks, threadsPerBlock>>>(V, E, dVisitLevelBw, dVisitLevelFw,nodeToBeVisited_d);
    cudaDeviceSynchronize();

    bool finished;
    bool* dFinished;
    cudaMalloc(&dFinished,sizeof(bool) *(1));
    do {
      finished = true;
      cudaMemcpy(dFinished, &finished, sizeof(bool) * (1), cudaMemcpyHostToDevice);
      fwd_pass<<<numBlocks,threadsPerBlock>>>(V, csr.d_fr_ptr, csr.d_fc_ptr, dFinished, dVisitLevelFw, dVisit, dVisitFw); ///DONE from varList
      cudaDeviceSynchronize();
      cudaMemcpy(&finished, dFinished, sizeof(bool) * (1), cudaMemcpyDeviceToHost);

    }while(!finished);

    bool finished2;
    bool* dFinished2;
    cudaMalloc(&dFinished2,sizeof(bool) *(1));
    do {
      finished2 = true;
      cudaMemcpy(dFinished2, &finished2, sizeof(bool) * (1), cudaMemcpyHostToDevice);
      fwd_pass2<<<numBlocks,threadsPerBlock>>>(V, csr.d_br_ptr, csr.d_bc_ptr, dFinished2, dVisitLevelBw, dVisit, dVisitFw, dVisitBw); ///DONE from varList
      cudaDeviceSynchronize();
      cudaMemcpy(&finished2, dFinished2, sizeof(bool) * (1), cudaMemcpyDeviceToHost);
    }while(!finished2);

    int sccsize = 0; int*d_sccsize; cudaMalloc(&d_sccsize,sizeof(int));
    cudaMemcpy(d_sccsize, &sccsize, sizeof(int), cudaMemcpyHostToDevice);
    
    scc_fw_kernel4<<<numBlocks, threadsPerBlock>>>(V, E, dVisit, dVisitFw, dVisitBw,d_sccsize,scc,scc_offset,noNewComp_d,sccindex_d);
    cudaDeviceSynchronize();

    cudaMemcpy(&sccsize, d_sccsize, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&noNewComp, noNewComp_d, sizeof(bool), cudaMemcpyDeviceToHost);

    if(noNewComp == false)
    {
      int*address = scc_offset + sccoffset;
      cudaMemcpy(address,&sccsize,sizeof(int),cudaMemcpyHostToDevice);
      sccCount = sccCount + 1;
      sccoffset = sccoffset + 1;
    }
    
    cudaMemcpy(&noNewComp, noNewComp_d, sizeof(bool), cudaMemcpyDeviceToHost);

  } 
  cudaFree(dVisitLevelBw);
  cudaFree(dVisitLevelFw);
  cudaFree(dVisitBw);
  cudaFree(dVisitFw);
  cudaFree(dVisit);
  cudaFree(dOutDeg);
  cudaFree(dInDeg);
  printf("Scc Count is %d\n", sccCount);
}

int main(int argc, char* argv[]) {
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename> <float_value>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    vector<pair<int,int>>list_pair;
    ifstream inputFile(filename); 
    string line;
    int V, E;
    if (inputFile.is_open()) {
        if (getline(inputFile, line)) {
            stringstream ss(line);
            if (ss >> V >> E) {
                cout << "Vertices count: " << V << endl;
                cout << "Edges count: " << E << endl;
            } else {
                cout << "Error reading V and E from the first line" << endl;
                return 1;
            }
        }

        while (getline(inputFile, line)) {
            stringstream ss(line);
            int firstVertex, secondVertex;
            char comma; 

            if (ss >> firstVertex >> comma >> secondVertex) {
                list_pair.push_back({firstVertex, secondVertex});
            }
        }
        inputFile.close();
    } else {
        cout << "Unable to open file" << endl;
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    csr_format csr(V,list_pair.size());
    csr.create_csr_format(list_pair);

    thrust::device_vector<int>d_scc(V,0);
    thrust::device_vector<int>d_scc_offset(V+1,0);

    int* sccptr = thrust::raw_pointer_cast(d_scc.data());
    int*offsetptr = thrust::raw_pointer_cast(d_scc_offset.data());

    SCC(csr,sccptr,offsetptr);

    thrust::exclusive_scan(d_scc_offset.begin(),d_scc_offset.begin()+V+1,d_scc_offset.begin());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken is %f\n",milliseconds/1000);
    csr.free_csr_format();
    return 0;
}