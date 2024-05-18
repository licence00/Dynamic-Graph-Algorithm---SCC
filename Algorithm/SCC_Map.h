#ifndef SCC_Map_H
#define SCC_Map_H

#include "SCC.h"
#define BLOCKSIZE 1024

__global__ void fill_the_rs(edge**edges,int edgesize,int label_node,int*fr,int*br)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<edgesize)
  {
    if(edges[i]->label == label_node)
    {
      atomicAdd(&fr[edges[i]->vertices[0]],1);
      atomicAdd(&br[edges[i]->vertices[1]],1);
    }
  }
}

__global__ void fill_the_rs1(int2*edgelist,int edgecount,int*fr,int*br)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<edgecount)
  {
    atomicAdd(&fr[edgelist[i].x],1);
    atomicAdd(&br[edgelist[i].y],1);
  }
}

__global__ void fill_the_cs(edge**edges,int edgesize,int label,int*fr,int*fc,int*br,int*bc,int*temp_f,int*temp_b)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<edgesize)
  {
    if(edges[i]->label == label)
    {
      int v1 = edges[i]->vertices[0];
      int v2 = edges[i]->vertices[1];
      fc[fr[v1]+atomicAdd(&temp_f[v1],1)] = v2;
      bc[br[v2]+atomicAdd(&temp_b[v2],1)] = v1;
    }
  }
}

__global__ void fill_the_cs1(int2*edgelist,int edgecount,int*fr,int*fc,int*br,int*bc,int*temp_f,int*temp_b)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<edgecount)
  {
    int v1 = edgelist[i].x;
    int v2 = edgelist[i].y;
    fc[fr[v1]+atomicAdd(&temp_f[v1],1)] = v2;
    bc[br[v2]+atomicAdd(&temp_b[v2],1)] = v1;
  }
}

/*Creating the csr format for the condense form of a graph in order to apply scc parallelized algorithm*/
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
    this->RSize = vertices+2;
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
  void create_csr_format(edge**edges,int edgesize,int label)
  {
    long int blocks = (edgesize + BLOCKSIZE - 1) / BLOCKSIZE;

    fill_the_rs<<<blocks,BLOCKSIZE>>>(edges,edgesize,label,d_fr_ptr,d_br_ptr);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error rs : " << cudaGetErrorString(err) << std::endl;
        return;
    }

    thrust::exclusive_scan(Fr.begin(),Fr.end(),Fr.begin());
    thrust::exclusive_scan(Br.begin(),Br.end(),Br.begin());

    thrust::device_vector<int>temp_f(this->Vertices,0);
    thrust::device_vector<int>temp_b(this->Vertices,0);

    int*temp_f_ptr = thrust::raw_pointer_cast(temp_f.data());
    int*temp_b_ptr = thrust::raw_pointer_cast(temp_b.data());

    fill_the_cs<<<blocks,BLOCKSIZE>>>(edges,edgesize,label,d_fr_ptr,d_fc_ptr,d_br_ptr,d_bc_ptr,temp_f_ptr,temp_b_ptr);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error cs : " << cudaGetErrorString(err) << std::endl;
        return;
    }

  }
  void create_csr_format1(int2*edgelist,int edgecount)
  {
    int blocks = (edgecount + BLOCKSIZE - 1) / BLOCKSIZE;

    fill_the_rs1<<<blocks,BLOCKSIZE>>>(edgelist,edgecount,d_fr_ptr,d_br_ptr);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error rs1 : " << cudaGetErrorString(err) << std::endl;
        return;
    }

    thrust::exclusive_scan(Fr.begin(),Fr.end(),Fr.begin());
    thrust::exclusive_scan(Br.begin(),Br.end(),Br.begin());

    thrust::device_vector<int>temp_f(this->Vertices,0);
    thrust::device_vector<int>temp_b(this->Vertices,0);

    int*temp_f_ptr = thrust::raw_pointer_cast(temp_f.data());
    int*temp_b_ptr = thrust::raw_pointer_cast(temp_b.data());

    fill_the_cs1<<<blocks,BLOCKSIZE>>>(edgelist,edgecount,d_fr_ptr,d_fc_ptr,d_br_ptr,d_bc_ptr,temp_f_ptr,temp_b_ptr);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error rs2 : " << cudaGetErrorString(err) << std::endl;
        return;
    }
  }
  void clear_csr_format()
  {
    Fr.clear();
    Fc.clear();
    Br.clear();
    Bc.clear();
  }
};


//status is true if we are in master node
void SCC(csr_format*csr,int*scc,int*scc_offset,int*scc_count,bool status)
{
  int V = csr->Vertices;
  int E = csr->Edges;
  int numThreads = BLOCKSIZE;
  int numBlocks = (V + BLOCKSIZE - 1) / BLOCKSIZE;
  int threadsPerBlock = BLOCKSIZE;

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
  
  while(!noNewComp) 
  {
    
    noNewComp = true;
    cudaMemcpy(noNewComp_d, &noNewComp, sizeof(bool), cudaMemcpyHostToDevice);
    bool noNewNode = false;

    
    while(!noNewNode) 
    {
      noNewNode = true;
      cudaMemcpy(noNewNode_d, &noNewNode, sizeof(bool), cudaMemcpyHostToDevice);
      cudaMemcpy(sccCount_d, &sccCount, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(sccoffset_d, &sccoffset, sizeof(int), cudaMemcpyHostToDevice);
      scc_fw_kernel0<<<numBlocks, threadsPerBlock>>>(V, E, csr->d_fr_ptr, csr->d_fc_ptr, csr->d_bc_ptr,csr->d_br_ptr, dVisit, dOutDeg, dInDeg,scc,scc_offset,sccCount_d,sccindex_d,sccoffset_d,noNewNode_d,status);
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

      fwd_pass<<<numBlocks,threadsPerBlock>>>(V, csr->d_fr_ptr, csr->d_fc_ptr, dFinished, dVisitLevelFw, dVisit, dVisitFw); 
      cudaDeviceSynchronize();
      cudaMemcpy(&finished, dFinished, sizeof(bool) * (1), cudaMemcpyDeviceToHost);

    }while(!finished);

    bool finished2;
    bool* dFinished2;
    cudaMalloc(&dFinished2,sizeof(bool) *(1));
    do {
      finished2 = true;
      cudaMemcpy(dFinished2, &finished2, sizeof(bool) * (1), cudaMemcpyHostToDevice);

      fwd_pass2<<<numBlocks,threadsPerBlock>>>(V, csr->d_br_ptr, csr->d_bc_ptr, dFinished2, dVisitLevelBw, dVisit, dVisitFw, dVisitBw); 
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
  cudaMemcpy(scc_count,&sccCount,sizeof(int),cudaMemcpyHostToDevice);
  csr->clear_csr_format();
}

__global__ void kernel1(node*current,int nodesize,int2*ptr)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid<nodesize)
  {
    ptr[tid] = make_int2(current->nodes[tid],tid);
  }
}

__global__ void kernel2(edge**edges,int total_edges,node*current,int*d_edges_count)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < total_edges)
  {
    if(edges[tid]->label == current->label)
    {
      atomicAdd(d_edges_count,1);
    }
  }
}

__global__ void kernel3(edge**edges,int edgesize,node*current,int2*edge_list,int2*mapping,int mapsize,int*t_index)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < edgesize)
  {
    if(edges[tid]->label == current->label)
    {
      int id = atomicAdd(t_index,1);
      for(int i=0;i<mapsize;i++)
      {
        if(mapping[i].x == edges[tid]->vertices[0])
        {
          edge_list[id].x = mapping[i].y;
        }
        if(mapping[i].x == edges[tid]->vertices[1])
        {
          edge_list[id].y = mapping[i].y;
        }
      }
    }
  }
}

__global__ void kernel4(node*current,int scc_sizes,int*scc)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < current->nodes_size)
  {
    scc[tid] = current->nodes[scc[tid]];
  }
}

__global__ void get_induce_vertex(node*org,int*val)
{
  *val = org->induce_vertex;
}

/*
  -->Creating a Map to Vertices and set them in a range(0,V) as there can be negative vertices 
  -->Applying the standard SCC mapping on the mapped vertices and edges
*/

void SCC_With_Mapping(edge**edges,int edgesize,node*current,int h_nodes_size,int*scc,int*scc_offset,int*scc_count,int vertices,bool status)
{   
    //status is true its master node
    int mapsize;
    if(status){mapsize = h_nodes_size;}
    else{mapsize = h_nodes_size+1;}

    thrust::device_vector<int2>vertex_to_index(mapsize);
    int2*vertex_to_index_ptr = thrust::raw_pointer_cast(vertex_to_index.data());

    //Creating a Map to the current vertices with the vertices in range(0,V)
    int blocks = (h_nodes_size + BLOCKSIZE - 1) / BLOCKSIZE;
    kernel1<<<blocks,BLOCKSIZE>>>(current,h_nodes_size,vertex_to_index_ptr);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error map : " << cudaGetErrorString(err) << std::endl;
        return;
    }

    //Mapping for the induce vertex in the Condensed graph
    int*d_magic_index; cudaMalloc(&d_magic_index,sizeof(int));
    get_induce_vertex<<<1,1>>>(current,d_magic_index); cudaDeviceSynchronize(); 

    int magic_index; cudaMemcpy(&magic_index,d_magic_index,sizeof(int),cudaMemcpyDeviceToHost);
    if(magic_index < 0){magic_index = (-1)*(magic_index)+vertices;}
    else{magic_index = (magic_index)+vertices;}
    
    if(!status)
    {
      vertex_to_index[h_nodes_size] = make_int2(magic_index,h_nodes_size);
    }

    int h_edges_count = 0; int*d_edges_count; cudaMalloc(&d_edges_count,sizeof(int));
    cudaMemcpy(d_edges_count,&h_edges_count,sizeof(int),cudaMemcpyHostToDevice);

    // int total_edges = edgesize;
    blocks = (edgesize + BLOCKSIZE - 1) / BLOCKSIZE;

    kernel2<<<blocks,BLOCKSIZE>>>(edges,edgesize,current,d_edges_count);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error count : " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaMemcpy(&h_edges_count,d_edges_count,sizeof(int),cudaMemcpyDeviceToHost);

    /*********************/
    // cudaMemcpy(edges_count,&h_edges_count,sizeof(int),cudaMemcpyHostToDevice);
    // int*d_edges_index; cudaMalloc(&d_edges_index,sizeof(int)*h_edges_count);

    if(h_edges_count>0)
    {
      thrust::device_vector<int2>edgeList(h_edges_count);
      int2*edgeList_ptr = thrust::raw_pointer_cast(edgeList.data());
      int*t_index; cudaMalloc(&t_index,sizeof(int)); cudaMemset(t_index,0,sizeof(int));

      kernel3<<<blocks,BLOCKSIZE>>>(edges,edgesize,current,edgeList_ptr,vertex_to_index_ptr,mapsize,t_index);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
          std::cerr << "CUDA Error map edges : " << cudaGetErrorString(err) << std::endl;
          return;
      }

      // cudaMemcpy(edges_index,d_edges_index,sizeof(int)*h_edges_count,cudaMemcpyDeviceToHost);

        csr_format*csr = new csr_format(mapsize,h_edges_count);        
        csr->create_csr_format1(edgeList_ptr,h_edges_count);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error csr fomrat1 : " << cudaGetErrorString(err) << std::endl;
            return;
        }
        

        SCC(csr,scc,scc_offset,scc_count,status);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error scc : " << cudaGetErrorString(err) << std::endl;
            return;
        }

        //Dereferencing back the vertices to the original vertices 
        blocks = (h_nodes_size + BLOCKSIZE - 1) / BLOCKSIZE;
        kernel4<<<blocks,BLOCKSIZE>>>(current,h_nodes_size,scc);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error de-map : " << cudaGetErrorString(err) << std::endl;
            return;
        }

        free(csr);
        edgeList.clear();
      }
    vertex_to_index.clear();
    
}

#endif