#ifndef SCC_DECREMENTAL_H
#define SCC_DECREMENTAL_H

#include "SCC.h"
#include "SCC_Map.h"
#include "SCC_Tree.h"
#include <iostream>
#include <fstream>


typedef struct 
{
  node*current;
  SCC_tree*T;
} subtree;

__global__ void find_address(node*nodes_ptr,int n,int nodeval1,int nodeval2,node**address1,node**address2)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n)
    {
        if(nodes_ptr[tid].label == nodeval1)
        {
            *address1 = nodes_ptr+tid;
        }
        if(nodes_ptr[tid].label == nodeval2)
        {
            *address2 = nodes_ptr+tid;
        }
    }
}

__global__ void find_label_address(int label,node*nodes_ptr,node**address)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < map_index)
    {
      if(nodes_ptr[tid].label == label)
      {
        *address = nodes_ptr+tid;
      }
    }
}

/*Finds the least common ancestor if two nodes address are given */
__global__ void findLCA(node** node_1, node** node_2, node** LCA,node*master_node,bool*status)
{
    node*node1 = *(node_1); node*node2 = *(node_2);
    int depth1 = node1->depth;
    int depth2 = node2->depth;

    while (depth1 > depth2) {
        node1 = node1->parent;
        depth1--;
    }
    while (depth2 > depth1) {
        node2 = node2->parent;
        depth2--;
    }

    while (node1 != node2) {
        node1 = node1->parent;
        node2 = node2->parent;
		    if(node1 == nullptr || node2 == nullptr)
        {
            *LCA = master_node; *status = true;
            return;
        }
    }
    *status = false;
    *LCA = node1; //LCA is the common ancestor
}

__global__ void reach_fwd_pass(int N, int* dOffsetArray, int* dEdgelist, bool* dFinished, int* dVisit,bool*reach)
{
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= N) return;
  if (dVisit[v] == 1 && reach[v] == false)
  {
    for (int edge = dOffsetArray[v]; edge < dOffsetArray[v+1]; edge++) {  
      int w = dEdgelist[edge];
      if (dVisit[w] == 0) 
      {
        dVisit[w] = 1;
        *dFinished = false;
      } 
    } 
    reach[v] = true;
  } 
}

__global__ void reach_kernel1(node*current,int nodesize,int2*ptr,int*begin)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < nodesize)
  {
    if(current->nodes[tid] == current->induce_vertex)
    {
      *begin = tid;
    }
    ptr[tid] = make_int2(current->nodes[tid],tid);
  }
}

__global__ void reach_kernel2(int V,int* dVisitLevelFWS, int* dVisitLevelFWE,int*begin,int end)
{
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (v == *begin) 
  {
    dVisitLevelFWS[v] = 1;
  } 
  if (v == end) 
  {
    dVisitLevelFWE[v] = 1;
  } 
}

__global__ void reach_kernel3(int N, int* dVisitLevelFWS, int* dVisitLevelFWE,node*current,int nodesize,int*reach_ptr,int*reach_size,int*unreach_ptr,int*unreach_size)
{
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= N) return;
  if((dVisitLevelFWS[v] == 0 || dVisitLevelFWE[v] == 0) && v != nodesize) //excluding the induce vertex's map index 
  {
    unreach_ptr[atomicAdd(unreach_size,1)] = current->nodes[v];
  } 
  if(dVisitLevelFWS[v] == 1 && dVisitLevelFWE[v] == 1 && v != nodesize) //excluding the induce vertex's map index
  {
    reach_ptr[atomicAdd(reach_size,1)] = current->nodes[v];
  }
}

__global__ void d_kernel1(edge**edges,int total_edges,node*current,int*count)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < total_edges)
  {
    if(edges[tid]->label == current->label)
    {
      atomicAdd(count,1);
    }
  }
}

__global__ void d_kernel2(edge**edges,int edgesize,node*current,int2*edge_list,int2*mapping,int mapsize,int*d_index)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < edgesize)
  {
    if(edges[tid]->label == current->label)
    {
      int id = atomicAdd(d_index,1);
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


void reachability(edge**edges,int edgesize,node*current,int h_nodes_size,int*reach_ptr,int*reach_size,int*unreach_ptr,int*unreach_size,int V)
{
    int mapsize = h_nodes_size+1;

    thrust::device_vector<int2>vertex_to_index(mapsize);
    int2*vertex_to_index_ptr = thrust::raw_pointer_cast(vertex_to_index.data());

    int*begin; cudaMalloc(&begin,sizeof(int));
    int blocks = (h_nodes_size + BLOCKSIZE - 1 ) / BLOCKSIZE;
    reach_kernel1<<<blocks,BLOCKSIZE>>>(current,h_nodes_size,vertex_to_index_ptr,begin);
    cudaDeviceSynchronize();

    //Mapping for the induce vertex in the Condensed graph
    int*d_magic_index; cudaMalloc(&d_magic_index,sizeof(int));
    get_induce_vertex<<<1,1>>>(current,d_magic_index); cudaDeviceSynchronize(); 

    int magic_index; cudaMemcpy(&magic_index,d_magic_index,sizeof(int),cudaMemcpyDeviceToHost);
    if(magic_index < 0){magic_index = (-1)*(magic_index)+V;}
    else{magic_index = (magic_index)+V;}

    vertex_to_index[h_nodes_size] = make_int2(magic_index,h_nodes_size);

    int h_count_edges = 0; int*d_count_edges; cudaMalloc(&d_count_edges,sizeof(int));
    cudaMemcpy(d_count_edges,&h_count_edges,sizeof(int),cudaMemcpyHostToDevice);

    blocks = (edgesize + BLOCKSIZE - 1 ) / BLOCKSIZE;

    d_kernel1<<<blocks,BLOCKSIZE>>>(edges,edgesize,current,d_count_edges);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_count_edges,d_count_edges,sizeof(int),cudaMemcpyDeviceToHost);

    thrust::device_vector<int2>edgeList(h_count_edges);
    int2*edgeList_ptr = thrust::raw_pointer_cast(edgeList.data());

    int*d_index; cudaMalloc(&d_index,sizeof(int)); cudaMemset(d_index,0,sizeof(int));
    d_kernel2<<<blocks,BLOCKSIZE>>>(edges,edgesize,current,edgeList_ptr,vertex_to_index_ptr,mapsize,d_index);
    cudaDeviceSynchronize();
    
    csr_format csr(mapsize,h_count_edges);
    csr.create_csr_format1(edgeList_ptr,h_count_edges);

    thrust::device_vector<int>d_VisitFWS(mapsize,0); int*VisitFWS = thrust::raw_pointer_cast(d_VisitFWS.data());
    thrust::device_vector<int>d_VisitFWE(mapsize,0); int*VisitFWE = thrust::raw_pointer_cast(d_VisitFWE.data());

    thrust::device_vector<bool>f_reach(mapsize,false); bool*f_reach_ptr = thrust::raw_pointer_cast(f_reach.data());
    thrust::device_vector<bool>b_reach(mapsize,false); bool*b_reach_ptr = thrust::raw_pointer_cast(b_reach.data());
    int numBlocks = (mapsize + BLOCKSIZE - 1) / BLOCKSIZE;

    reach_kernel2<<<numBlocks,BLOCKSIZE>>>(mapsize, VisitFWS, VisitFWE, begin, h_nodes_size);
    cudaDeviceSynchronize();

    bool finished; bool* dFinished; cudaMalloc(&dFinished,sizeof(bool) *(1));
    do {
      finished = true;
      cudaMemcpy(dFinished, &finished, sizeof(bool) * (1), cudaMemcpyHostToDevice);

      reach_fwd_pass<<<numBlocks,BLOCKSIZE>>>(mapsize, csr.d_fr_ptr, csr.d_fc_ptr, dFinished, VisitFWS, f_reach_ptr); 
      cudaDeviceSynchronize();
      cudaMemcpy(&finished, dFinished, sizeof(bool) * (1), cudaMemcpyDeviceToHost);

    }while(!finished);

    bool finished2; bool* dFinished2; cudaMalloc(&dFinished2,sizeof(bool) *(1));
    do {
      finished2 = true;
      cudaMemcpy(dFinished2, &finished2, sizeof(bool) * (1), cudaMemcpyHostToDevice);

      reach_fwd_pass<<<numBlocks,BLOCKSIZE>>>(mapsize, csr.d_br_ptr, csr.d_bc_ptr, dFinished2, VisitFWE, b_reach_ptr);
      cudaDeviceSynchronize();
      cudaMemcpy(&finished2, dFinished2, sizeof(bool) * (1), cudaMemcpyDeviceToHost);
    }while(!finished2);

    reach_kernel3<<<numBlocks,BLOCKSIZE>>>(mapsize, VisitFWS, VisitFWE, current, h_nodes_size,reach_ptr,reach_size,unreach_ptr,unreach_size);
    cudaDeviceSynchronize();

    edgeList.clear(); vertex_to_index.clear(); f_reach.clear(); b_reach.clear(); d_VisitFWS.clear(); d_VisitFWE.clear();
}

__global__ void find_id_edge(int u,int v,edge**edges,int edgesize,node**current,int*id_edge,int V,bool*status)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < edgesize)
    {
      if(edges[index]->label == (*current)->label)
      {
        int f_vertex = edges[index]->vertices[0]; int s_vertex = edges[index]->vertices[1];
        if(status)
        {
          if((*current)->induce_vertex < 0 && s_vertex == (-1)*((*current)->induce_vertex)+V){s_vertex = (*current)->induce_vertex;}
          else if((*current)->induce_vertex >=0 && s_vertex == (*current)->induce_vertex+V){s_vertex = (*current)->induce_vertex;}
        }
        if(f_vertex>=0 && s_vertex>=0 && f_vertex == u && s_vertex == v)
        {
            *id_edge = index;
        }
        else if(f_vertex<0 && s_vertex>=0 && s_vertex == v && edges[index]->connect[0] == u)
        {
            *id_edge = index;
        }
        else if(f_vertex>=0 && s_vertex<0 && f_vertex == u && edges[index]->connect[1] == v)
        {
            *id_edge = index;
        }
        else if(f_vertex<0 && s_vertex<0 && edges[index]->connect[0] == u && edges[index]->connect[1] == v)
        {
            *id_edge = index;
        }
      }
    }
}

__global__ void set_parent_address(node*current,node**parent,node*master_node,bool*master)
{
    if(current->parent != nullptr)
    {
      *parent = current->parent;
      *master = false;
    }
    else
    {
      *parent = master_node;
      *master = true;
    }
}

__global__ void store_the_nodes(node*current,int*store)
{
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < current->nodes_size)
    {
      store[tid] = current->nodes[tid];
    }
}

__global__ void find_label_index(int nodes_size,node*current,int*nodes,int*index)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < nodes_size)
    {
      if(nodes[tid] == current->label)
      {
        *index = tid;
      }
    }
}

void set_parent_size(node*parent,int parent_nodes_size,int unreach_size,bool node_zero)
{
  if(node_zero)
  {
    int new_size = parent_nodes_size-1; int*new_array;
    cudaMalloc(&new_array,sizeof(int)*(parent_nodes_size+unreach_size-1));
    cudaMemcpy(&(parent->nodes),&new_array,sizeof(int*),cudaMemcpyHostToDevice);
    cudaMemcpy(&(parent->nodes_size),&new_size,sizeof(int),cudaMemcpyHostToDevice);
  }
  else
  {
    int new_size = parent_nodes_size; int*new_array;
    cudaMalloc(&new_array,sizeof(int)*(parent_nodes_size+unreach_size));
    cudaMemcpy(&(parent->nodes),&new_array,sizeof(int*),cudaMemcpyHostToDevice);
    cudaMemcpy(&(parent->nodes_size),&new_size,sizeof(int),cudaMemcpyHostToDevice);
  }
}

__global__ void fill_parent_nodes_zero(node*current,int nodes_size,int*nodes,int*d_index)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < nodes_size)
  {
    if(tid < *d_index)
    {
      current->nodes[tid] = nodes[tid];
    }
    else if(tid > *d_index)
    {
      current->nodes[tid-1] = nodes[tid];
    }
  } 
}

__global__ void fill_current_nodes(node*current,int*reach_ptr,int reach_size)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < reach_size)
    {
      current->nodes[tid] = reach_ptr[tid];
    }
}

__global__ void fill_parent_nodes(node*current,int nodes_size,int*nodes)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < nodes_size)
    {
      current->nodes[tid] = nodes[tid];
    }
}

void additional_nodes_for_parent(node*parent,int parent_nodes_size,int unreach_size,node*current,bool node_zero)
{
  thrust::device_vector<int>store_nodes(parent_nodes_size); int*store_ptr = thrust::raw_pointer_cast(store_nodes.data());

  int blocks = (parent_nodes_size + BLOCKSIZE - 1) / BLOCKSIZE;

  store_the_nodes<<<blocks,BLOCKSIZE>>>(parent,store_ptr); cudaDeviceSynchronize();

  set_parent_size(parent,parent_nodes_size,unreach_size,node_zero);

  if(node_zero)
  {
    //parent nodes size should be the initial size because the store ptr size is initial size
    int*d_index; cudaMalloc(&d_index,sizeof(int));
    find_label_index<<<blocks,BLOCKSIZE>>>(parent_nodes_size,current,store_ptr,d_index); cudaDeviceSynchronize();

    fill_parent_nodes_zero<<<blocks,BLOCKSIZE>>>(parent,parent_nodes_size,store_ptr,d_index); cudaDeviceSynchronize();
  }
  else
  {
    fill_parent_nodes<<<blocks,BLOCKSIZE>>>(parent,parent_nodes_size,store_ptr); cudaDeviceSynchronize();
  }
}

__global__ void get_current_index_map(node*current,node*nodes_ptr,int*index,int mapsize)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < mapsize)
    {
      if(nodes_ptr[tid].label == current->label)
      {
        *index = tid;
      }
    }
}

__global__ void check_and_set_ivertex(node*current,node*parent,int nodes_zero)
{
    if(nodes_zero && parent->induce_vertex == current->label)
    {
      parent->induce_vertex = parent->nodes[0]; 
      //changing the induced vertex if the current node contains zero nodes then we will delete it
    }
}

__global__ void find_nodes_d(node*current,bool*flag)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < current->nodes_size)
    {
      if(current->nodes[tid] == current->induce_vertex)
      {
        *flag = true;
      }
    }
}

__global__ void set_new_ivertex(node*current,bool*flag)
{
  if(flag == false && current->nodes_size > 0)
  {
    current->induce_vertex = current->nodes[0];
  }
}

__global__ void change_parent(node*nodes_ptr,int mapsize,int index)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < mapsize)
    {
      int remove_label = (-1)*(nodes_ptr[index].label);
      if(tid>index && nodes_ptr[tid].parent != nullptr)
      {
        if((nodes_ptr[tid].parent->label)*(-1) > remove_label)
        {
          nodes_ptr[tid].parent = nodes_ptr[tid].parent - 1;
        }
      }
    }
}

void remove_label_from_map(node*current,SCC_tree*G,node*nodes_ptr)
{
    int index; int*d_index; cudaMalloc(&d_index,sizeof(int));
    int mapsize; cudaMemcpyFromSymbol(&mapsize,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);

    int blocks = (mapsize + BLOCKSIZE - 1) / BLOCKSIZE;
    get_current_index_map<<<blocks,BLOCKSIZE>>>(current,nodes_ptr,d_index,mapsize);
    cudaDeviceSynchronize();

    cudaMemcpy(&index,d_index,sizeof(int),cudaMemcpyDeviceToHost);

    blocks = (mapsize + BLOCKSIZE - 1) / BLOCKSIZE;
    change_parent<<<blocks,BLOCKSIZE>>>(nodes_ptr,mapsize,index); cudaDeviceSynchronize();

    thrust::copy(G->map.begin()+index+1,G->map.end(),G->map.begin()+index);   
    G->map.pop_back();
}


__global__ void update_current(node**current)
{
  if((*current)->parent != nullptr)
  {
    *current = (*current)->parent;
  }
  else
  {
    *current = nullptr;
  }
}

__global__ void change_stats_unreach(node*current,node*parent,int*unreach_ptr,int node_index,bool*flag,node*nodes_ptr,int mapsize)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < mapsize)
    {
      if(nodes_ptr[tid].label == unreach_ptr[node_index])
      {
        nodes_ptr[tid].parent = current->parent;
        if(current->parent != nullptr)
        {
          nodes_ptr[tid].depth = current->parent->depth + 1;
          nodes_ptr[tid].primary_ancestor = current->parent->primary_ancestor;
        }
        else
        {
          nodes_ptr[tid].depth = 0;
          nodes_ptr[tid].primary_ancestor = nodes_ptr[tid].label;
        }
        parent->nodes[atomicAdd(&parent->nodes_size,1)] = unreach_ptr[node_index];

        if(unreach_ptr[node_index] < 0)
        {
          *flag = true;
        }
      }
    }
}

__global__ void change_subtree_stats(node**current,node*nodes_ptr,int nodes_size,int node_index,int*d_label)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < map_index)
  {
    if(nodes_ptr[tid].label == (*current)->nodes[node_index])
    {
      nodes_ptr[tid].depth = (*current)->depth + 1;
      nodes_ptr[tid].primary_ancestor = (*current)->primary_ancestor;
      *d_label = nodes_ptr[tid].label;
    }
  }
}

__global__ void get_nodes_size_d(node**current,int*size)
{
  *size = (*current)->nodes_size;
}

void change_node_subtree(int un_index,node*nodes_ptr)
{
  queue<int>q; q.push(un_index);
  int mapsize; cudaMemcpyFromSymbol(&mapsize,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);

  int blocks = (mapsize + BLOCKSIZE - 1) / BLOCKSIZE; int label; int*d_label; cudaMalloc(&d_label,sizeof(int));

  while(!q.empty())
  {
    int node_index = q.front(); q.pop();
    node**current; cudaMalloc(&current,sizeof(node*));
    find_label_address<<<blocks,BLOCKSIZE>>>(node_index,nodes_ptr,current); cudaDeviceSynchronize();

    int h_nodes_size; int*d_nodes_size; cudaMalloc(&d_nodes_size,sizeof(int));
    get_nodes_size_d<<<1,1>>>(current,d_nodes_size); cudaDeviceSynchronize();
    cudaMemcpy(&h_nodes_size,d_nodes_size,sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<h_nodes_size;i++)
    {
      
      change_subtree_stats<<<blocks,BLOCKSIZE>>>(current,nodes_ptr,h_nodes_size,i,d_label); cudaDeviceSynchronize();
      
      cudaMemcpy(&label,d_label,sizeof(int),cudaMemcpyDeviceToHost);

      if(label < 0)
      {
        q.push(label);
      }
    }
  }
}

__global__ void adjust_parent_edges(edge**edges,int edgesize,node*parent,int*old_iv,int V,node*current,bool master,node*nodes_ptr)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x; 
  if(tid < edgesize)
  {
    if(edges[tid]->label == parent->label)
    {
      int f_vertex = edges[tid]->vertices[0]; int s_vertex = edges[tid]->vertices[1];
      if((*old_iv) < 0 && s_vertex == (-1)*(*old_iv)+V){s_vertex = (*old_iv);}
      else if((*old_iv) >= 0 && s_vertex == (*old_iv)+V){s_vertex = (*old_iv);}

      int connect; bool flag = false;
      if(f_vertex == current->label){connect = edges[tid]->connect[0]; flag = true;}
      else if(s_vertex == current->label){connect = edges[tid]->connect[1]; flag = true;}

      if(flag == true)
      {
        for(int index=0;index<map_index;index++)
        {
          if(nodes_ptr[index].label == connect)
          {
            node*pointer = nodes_ptr+index;
            if(!master)
            {
              while(pointer->parent != parent)
              {
                pointer = pointer->parent;
              }
              if(f_vertex == current->label)
              {
                f_vertex = pointer->label;
                if(f_vertex >= 0){edges[tid]->connect[0] = 0;}
              }
              else if(s_vertex == current->label)
              {
                s_vertex = pointer->label;
                if(s_vertex>=0){edges[tid]->connect[1] = 0;}
              }
              if(s_vertex>=0 && s_vertex == parent->induce_vertex){s_vertex = parent->induce_vertex+V;}
              else if(s_vertex<0 && s_vertex == parent->induce_vertex){s_vertex = (-1)*(parent->induce_vertex)+V;}
              edges[tid]->vertices[0] = f_vertex; edges[tid]->vertices[1] = s_vertex;
            }
            else
            {
              if(f_vertex == current->label)
              {
                f_vertex = pointer->primary_ancestor;
                if(f_vertex >= 0){edges[tid]->connect[0] = 0;}
              }
              else if(s_vertex == current->label)
              {
                s_vertex = pointer->primary_ancestor;
                if(s_vertex>=0){edges[tid]->connect[1] = 0;}
              }
              edges[tid]->vertices[0] = f_vertex; edges[tid]->vertices[1] = s_vertex; 
            }
            break;
          }
        }
      }
    }
  }
}

__global__ void adjust_current_edges(edge**edges,int edgesize,node*current,node*parent,int*old_iv,int V,bool master)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < edgesize)
    {
      if(edges[tid]->label == current->label)
      {
        int f_vertex = edges[tid]->vertices[0]; int s_vertex = edges[tid]->vertices[1];
        if((*old_iv) < 0 && s_vertex == (-1)*(*old_iv)+V){s_vertex = (*old_iv);}
        else if((*old_iv) >= 0 && s_vertex == (*old_iv)+V){s_vertex = (*old_iv);}

        bool flag1=false; bool flag2=false;
        for(int i=0;i<current->nodes_size;i++)
        {
          if(current->nodes[i] == f_vertex)
          {
            flag1 = true;
          }
          if(current->nodes[i] == s_vertex)
          {
            flag2 = true;
          }
        }
        if(!flag1 && !flag2)
        {
          if(!master)
          {
            if(s_vertex>=0 && s_vertex == parent->induce_vertex){s_vertex = parent->induce_vertex+V;}
            else if(s_vertex<0 && s_vertex == parent->induce_vertex){s_vertex = (-1)*(parent->induce_vertex)+V;}
          }
          edges[tid]->vertices[1] = s_vertex; edges[tid]->label = parent->label;
        }
        else if(flag1 && !flag2)
        {
          if(f_vertex>=0)
          {
            edges[tid]->connect[0] = f_vertex;
          }        
          if(!master)
          {
            if(s_vertex>=0 && s_vertex == parent->induce_vertex){s_vertex = parent->induce_vertex+V;}
            else if(s_vertex<0 && s_vertex == parent->induce_vertex){s_vertex = (-1)*(parent->induce_vertex)+V;}
          }
          edges[tid]->vertices[0] = current->label;  edges[tid]->vertices[1] = s_vertex;
          edges[tid]->label = parent->label;
        }
        else if(!flag1 && flag2)
        {
          int changed = current->label;
          if(!master && changed == parent->induce_vertex){changed = (-1)*(parent->induce_vertex)+V;}
          if(s_vertex>=0)
          {
            edges[tid]->connect[1] = s_vertex;
          }
          edges[tid]->vertices[1] = changed; edges[tid]->label = parent->label;
        }
        else if(flag1 && flag2)
        {
          if(s_vertex>=0 && s_vertex == current->induce_vertex){s_vertex = current->induce_vertex+V;}
          else if(s_vertex<0 && s_vertex == current->induce_vertex){s_vertex = (-1)*(current->induce_vertex)+V;}
          edges[tid]->vertices[1] = s_vertex;
        }
      }
    }  
}

void decremental(node**curr,SCC_tree*G)
{
    //Decremental code

    edge**edges = thrust::raw_pointer_cast(G->edges.data()); int edgesize = G->edges.size();
    int V = G->n_v;  node*nodes_ptr = thrust::raw_pointer_cast(G->map.data());

    node*current; cudaMemcpy(&current, curr, sizeof(node*), cudaMemcpyDeviceToHost);
    do
    {
      int h_nodes_size; int*d_nodes_size; cudaMalloc(&d_nodes_size,sizeof(int)); 
      get_nodes_size<<<1,1>>>(current,d_nodes_size); cudaDeviceSynchronize(); 
      cudaMemcpy(&h_nodes_size,d_nodes_size,sizeof(int),cudaMemcpyDeviceToHost);

      thrust::device_vector<int>d_reach(h_nodes_size);  int*reach_ptr = thrust::raw_pointer_cast(d_reach.data());
      thrust::device_vector<int>d_unreach(h_nodes_size);  int*unreach_ptr = thrust::raw_pointer_cast(d_unreach.data());

      int h_reach_size = 0; int*reach_size; cudaMalloc(&reach_size,sizeof(int));
      cudaMemcpy(reach_size,&h_reach_size,sizeof(int),cudaMemcpyHostToDevice);

      int h_unreach_size=0; int*unreach_size; cudaMalloc(&unreach_size,sizeof(int));
      cudaMemcpy(unreach_size,&h_unreach_size,sizeof(int),cudaMemcpyHostToDevice);

      //Apply reachability to find the reachable and unreachable nodes in the current node based on condensed graph
      reachability(edges,edgesize,current,h_nodes_size,reach_ptr,reach_size,unreach_ptr,unreach_size,V);

      cudaMemcpy(&h_reach_size,reach_size,sizeof(int),cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_unreach_size,unreach_size,sizeof(int),cudaMemcpyDeviceToHost);

      d_reach.resize(h_reach_size); d_unreach.resize(h_unreach_size);

      if(h_unreach_size>0)
      {
        int mapsize; set_nodes_size(current,h_reach_size);
        //Changing the current nodes based on the reachability
        if(h_reach_size>0)
        {
          int blocks = (h_reach_size + BLOCKSIZE - 1) / BLOCKSIZE;
          reach_ptr = thrust::raw_pointer_cast(d_reach.data());
          fill_current_nodes<<<blocks,BLOCKSIZE>>>(current,reach_ptr,h_reach_size); cudaDeviceSynchronize();
          cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess) {
              std::cerr << "CUDA Error fill current nodes: " << cudaGetErrorString(err) << std::endl;
              return;
          }
        }

        h_nodes_size = h_reach_size;

        bool node_zero = false;
        if(h_nodes_size == 0){node_zero = true;}

        //Set the parent address and get the node size of the parent
        node**d_parent; cudaMalloc(&d_parent,sizeof(node*)); bool*d_master; cudaMalloc(&d_master,sizeof(bool));
        set_parent_address<<<1,1>>>(current,d_parent,G->master_node,d_master); cudaDeviceSynchronize();

        bool master; node*parent; cudaMemcpy(&parent,d_parent,sizeof(node*),cudaMemcpyDeviceToHost);
        cudaMemcpy(&master,d_master,sizeof(bool),cudaMemcpyDeviceToHost);

        int h_parent_nodes_size; int*d_parent_nodes_size; cudaMalloc(&d_parent_nodes_size,sizeof(int));
        get_nodes_size<<<1,1>>>(parent,d_parent_nodes_size); cudaDeviceSynchronize();
        cudaMemcpy(&h_parent_nodes_size,d_parent_nodes_size,sizeof(int),cudaMemcpyDeviceToHost);

        //Fill the parent nodes with all the condition whether child contains one or zero nodes
        additional_nodes_for_parent(parent,h_parent_nodes_size,h_unreach_size,current,node_zero);

        cudaMemcpyFromSymbol(&mapsize,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);

        int blocks = (mapsize + BLOCKSIZE - 1) / BLOCKSIZE;
        nodes_ptr = thrust::raw_pointer_cast(G->map.data());

        for(int i=0;i<h_unreach_size;i++)
        {
          bool*flag; cudaMalloc(&flag,sizeof(bool)); cudaMemset(flag,false,sizeof(bool));
          unreach_ptr = thrust::raw_pointer_cast(d_unreach.data());
          change_stats_unreach<<<blocks,BLOCKSIZE>>>(current,parent,unreach_ptr,i,flag,nodes_ptr,mapsize);
          cudaDeviceSynchronize();
          cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess) {
              std::cerr << "CUDA Error change stats : " << cudaGetErrorString(err) << std::endl;
              return;
          }

          bool h_flag; cudaMemcpy(&h_flag,flag,sizeof(bool),cudaMemcpyDeviceToHost);
          if(h_flag)
          {
            int unreach_node; cudaMemcpy(&unreach_node,unreach_ptr+i,sizeof(int),cudaMemcpyDeviceToHost);
            change_node_subtree(unreach_node,nodes_ptr);
          }

        }

        int*d_old_p_iv; cudaMalloc(&d_old_p_iv,sizeof(int));
        get_induce_vertex<<<1,1>>>(parent,d_old_p_iv); cudaDeviceSynchronize();

        check_and_set_ivertex<<<1,1>>>(current,parent,node_zero); cudaDeviceSynchronize();

        blocks = (edgesize + BLOCKSIZE - 1) / BLOCKSIZE;
        adjust_parent_edges<<<blocks,BLOCKSIZE>>>(edges,edgesize,parent,d_old_p_iv,V,current,master,nodes_ptr); cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error parent edges : " << cudaGetErrorString(err) << std::endl;
            return;
        }

        int*old_iv; cudaMalloc(&old_iv,sizeof(int));
        get_induce_vertex<<<1,1>>>(current,old_iv); cudaDeviceSynchronize();

        if(h_nodes_size>0)
        {
          bool*flag; cudaMalloc(&flag,sizeof(bool)); cudaMemset(flag,false,sizeof(bool));
          blocks = (h_nodes_size + BLOCKSIZE - 1) / BLOCKSIZE;
          find_nodes_d<<<blocks,BLOCKSIZE>>>(current,flag); cudaDeviceSynchronize();
          
          set_new_ivertex<<<1,1>>>(current,flag); cudaDeviceSynchronize();
        }

        blocks = (edgesize + BLOCKSIZE - 1) / BLOCKSIZE;
        adjust_current_edges<<<blocks,BLOCKSIZE>>>(edges,edgesize,current,parent,old_iv,V,master);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error current edges : " << cudaGetErrorString(err) << std::endl;
            return;
        }

        update_current<<<1,1>>>(curr); cudaDeviceSynchronize();

        if(node_zero)
        {
          cudaMemcpyFromSymbol(&mapsize,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);
          remove_label_from_map(current,G,nodes_ptr);
          mapsize = mapsize - 1; //decrement the mapindex
          cudaMemcpyToSymbol(::map_index,&mapsize, sizeof(int) , 0 , cudaMemcpyHostToDevice);
        } 
        cudaMemcpy(&current, curr, sizeof(node*), cudaMemcpyDeviceToHost);
      }
      else
      {
        break;
      }
    }while(current!=nullptr);
}

__global__ void copy_edges(edge**edges,edge**temp_edges,int edgesize,int id_edge)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < edgesize)
    {
      if(tid < id_edge)
      {
        temp_edges[tid] = edges[tid];
      }
      else if(tid > id_edge)
      {
        temp_edges[tid-1] = edges[tid];
      }
    }
}

void Remove_Edge(int u,int v,SCC_tree*G)
{
    node*nodes_ptr = thrust::raw_pointer_cast(G->map.data());
    int mapsize; cudaMemcpyFromSymbol(&mapsize,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);

    //Remove the edge from the graph
    int blocks = (mapsize + BLOCKSIZE - 1) / BLOCKSIZE;
    //Finding the Least Common Ancestor
    node**address1; cudaMalloc(&address1,sizeof(node*));
    node**address2; cudaMalloc(&address2,sizeof(node*));

    find_address<<<blocks,BLOCKSIZE>>>(nodes_ptr,mapsize,u,v,address1,address2);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error find address: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    bool*d_status; cudaMalloc(&d_status,sizeof(bool));
    node**LCA; cudaMalloc(&LCA,sizeof(node*));
    findLCA<<<1,1>>>(address1,address2,LCA,G->master_node,d_status);
    cudaDeviceSynchronize();

    edge**edges = thrust::raw_pointer_cast(G->edges.data());
    int edgesize = G->edges.size(); int V = G->n_v;
    blocks = (edgesize + BLOCKSIZE - 1) / BLOCKSIZE;

    int id_edge; int*d_id_edge; cudaMalloc(&d_id_edge,sizeof(int)); cudaMemset(d_id_edge,-1,sizeof(int));
    find_id_edge<<<blocks,BLOCKSIZE>>>(u,v,edges,edgesize,LCA,d_id_edge,V,d_status);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error find edge: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaMemcpy(&id_edge,d_id_edge,sizeof(int),cudaMemcpyDeviceToHost);
    
    //Remove the edge from the edges list
    if(id_edge != -1)
    {
      int edgesize = G->edges.size();
      edge**temp_edges; cudaMalloc(&temp_edges,sizeof(edge*)*(edgesize-1));

      int blocks = (edgesize + BLOCKSIZE - 1) / BLOCKSIZE;
      copy_edges<<<blocks,BLOCKSIZE>>>(edges,temp_edges,edgesize,id_edge); cudaDeviceSynchronize();

      G->edges.resize(edgesize-1);

      edge**edges_ptr = thrust::raw_pointer_cast(G->edges.data());
      cudaMemcpy(edges_ptr,temp_edges,sizeof(edge*)*(edgesize-1),cudaMemcpyDeviceToDevice);

      bool status; cudaMemcpy(&status,d_status,sizeof(bool),cudaMemcpyDeviceToHost);

      if(!status) //If the edge is not removed from master node
      {
        decremental(LCA,G);
      }
    }
    else
    {
      printf("Edge not found in graph\n");
      exit(0);
    }
}


// __global__ void check_both_nodes(node**address,node**address1,int*ans)
// {
//     if((*address)->primary_ancestor == (*address1)->primary_ancestor)
//     {
//         *ans = 1;
//     }
//     else
//     {
//         *ans = 0;
//     }
// }
// void Query(int u,int v,SCC_tree*tree,int*nans)
// {
//     node*nodes_ptr = thrust::raw_pointer_cast(tree->map.data());
//     int mapsize; cudaMemcpyFromSymbol(&mapsize,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);
//     int blocks = (mapsize+BLOCKSIZE-1)/BLOCKSIZE;
//     node**address1; cudaMalloc(&address1,sizeof(node*));
//     node**address2; cudaMalloc(&address2,sizeof(node*));
//     find_address<<<blocks,BLOCKSIZE>>>(nodes_ptr,mapsize,u,v,address1,address2); cudaDeviceSynchronize();
//     int ans; int*d_ans; cudaMalloc(&d_ans,sizeof(int));
//     check_both_nodes<<<1,1>>>(address1,address2,d_ans);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&ans,d_ans,sizeof(int),cudaMemcpyDeviceToHost);
//     *nans = ans;
// }

#endif // SCC_DECREMENTAL_H