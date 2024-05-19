#ifndef SCC_INCREMENTAL_H
#define SCC_INCREMENTAL_H

#include "SCC_Map.h"
#include "SCC_Decremental.h"
#include <cuda.h>

// __global__ void find_address(node*nodes_ptr,int n,int nodeval1,int nodeval2,node**address1,node**address2)
// {
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if(tid < n)
//     {
//         if(nodes_ptr[tid].label == nodeval1)
//         {
//             *address1 = nodes_ptr+tid;
//         }
//         if(nodes_ptr[tid].label == nodeval2)
//         {
//             *address2 = nodes_ptr+tid;
//         }
//     }
// }

/*Finds the least common ancestor if two nodes address are given */
__global__ void findLCA(node** node_1, node** node_2, node** LCA,int* f,int* s,bool* status,int* label,int* iv,node* master_node)
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

    node*prev1 = node1; node*prev2 = node2;

    while (node1 != node2) 
    {
        prev1 = node1; prev2 = node2;
        node1 = node1->parent;
        node2 = node2->parent;
		    if(node1 == nullptr || node2 == nullptr)
        {
          //Not found the LCA therefore it is master node
          *LCA = master_node; *f = prev1->primary_ancestor; *s = prev2->primary_ancestor;
          *label = master_node->label; *status=true;
          return;
        }
    }
    //Found the LCA
    *LCA = node1; *f = prev1->label; *s = prev2->label;
    *label = node1->label; *iv = node1->induce_vertex; *status = false;
}

__global__ void get_label(node*org,int*d_label)
{
  *d_label = org->label;
}

__global__ void assign_single_nodes(node*current,int*scc,int*scc_offset,int*scc_count,int*new_nodes_count,bool status,int V)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < *scc_count)
  {
    int scc_size = scc_offset[tid+1] - scc_offset[tid];
    if(scc_size > 1) 
    {
      if(scc_size != current->nodes_size || status) //status is true if current is master node
      {
        atomicAdd(new_nodes_count,1);
      }
    }
    else if(scc_size == 1 && scc[scc_offset[tid]] < V)
    {
      current->nodes[atomicAdd(&(current->nodes_size),1)] = scc[scc_offset[tid]];
    }
  }
}

__global__ void fill_new_nodes(int*sccindex,node*current,int*scc,int*scc_offset,int*new_nodes_count,bool status,node*nodes_ptr)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < *new_nodes_count)
  {
    int sccsize = scc_offset[sccindex[tid]+1] - scc_offset[sccindex[tid]];
    for(int index=0;index<sccsize;index++)
    {
      nodes_ptr[map_index+tid].nodes[index] = scc[scc_offset[sccindex[tid]]+index];
    }
    nodes_ptr[map_index+tid].label = (-1)*atomicAdd(&label_count,1);
    nodes_ptr[map_index+tid].induce_vertex = nodes_ptr[map_index+tid].nodes[0];
    nodes_ptr[map_index+tid].nodes_size = sccsize;
    if(status) //case of master node
    {
      nodes_ptr[map_index+tid].parent = nullptr;
      nodes_ptr[map_index+tid].depth = 0;
      nodes_ptr[map_index+tid].primary_ancestor = nodes_ptr[map_index+tid].label;
    }
    else
    {
      nodes_ptr[map_index+tid].parent = current;
      nodes_ptr[map_index+tid].depth = current->depth+1;
      nodes_ptr[map_index+tid].primary_ancestor = current->primary_ancestor;
    }
    current->nodes[atomicAdd(&(current->nodes_size),1)] = nodes_ptr[map_index+tid].label;
  }
}


__global__ void set_nn_parent_address(node*current,int mapsize,node*nodes_ptr,int index)
{ 
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < mapsize)
  {
    if(nodes_ptr[tid].label == current->nodes[index])
    {
      nodes_ptr[tid].parent = current;
    }
  }
}

void set_nn_nodes_parent(node*nn,node*label)
{
  int h_nn_nodes; int*d_nn_nodes; cudaMalloc(&d_nn_nodes,sizeof(int));
  get_nodes_size<<<1,1>>>(nn,d_nn_nodes); cudaDeviceSynchronize();
  cudaMemcpy(&h_nn_nodes,d_nn_nodes,sizeof(int),cudaMemcpyDeviceToHost);

  int map_size; cudaMemcpyFromSymbol(&map_size,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);
  int blocks = (map_size + BLOCKSIZE - 1)/ BLOCKSIZE;

  for(int i=0;i<h_nn_nodes;i++)
  {
    set_nn_parent_address<<<blocks,BLOCKSIZE>>>(nn,map_size,label,i); //current is the new node address
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error parent setting: " << cudaGetErrorString(err) << std::endl;
        return;
    }
  }
}

void incremental(node**curr,SCC_tree*G,int status)
{
    //Incremental code
    edge**edges = thrust::raw_pointer_cast(G->edges.data()); int edgesize = G->edges.size();
    int V = G->n_v; node*nodes_ptr = thrust::raw_pointer_cast(G->map.data());
    //push the value present in node curr into the queue
    node*temp_curr; cudaMemcpy(&temp_curr,curr,sizeof(node*),cudaMemcpyDeviceToHost);

    queue<node*>q; q.push(temp_curr); bool i_status = status;

    while(!q.empty())
    {
      node*current = q.front(); q.pop();

      int depth; int*d_depth; cudaMalloc(&d_depth,sizeof(int));
      depth_of_node<<<1,1>>>(current,d_depth); cudaDeviceSynchronize();
      cudaMemcpy(&depth,d_depth,sizeof(int),cudaMemcpyDeviceToHost);

      if(depth>11)
      {
        //Do Nothing as we are imposing a limit on the depth of the tree
      }
      else
      {
        int h_nodes_size; int*d_nodes_size; cudaMalloc(&d_nodes_size,sizeof(int)); 
        get_nodes_size<<<1,1>>>(current,d_nodes_size); cudaDeviceSynchronize(); 
        cudaMemcpy(&h_nodes_size,d_nodes_size,sizeof(int),cudaMemcpyDeviceToHost);

        thrust::device_vector<int>scc(h_nodes_size,0); thrust::device_vector<int>scc_offset(h_nodes_size+1,0);

        //Raw pointers for the SCC arrays
        int*d_scc_ptr = thrust::raw_pointer_cast(scc.data());
        int*d_scc_offset_ptr = thrust::raw_pointer_cast(scc_offset.data());
        int*d_scc_count; cudaMalloc(&d_scc_count,sizeof(int));
          
        SCC_With_Mapping(edges,edgesize,current,h_nodes_size,d_scc_ptr,d_scc_offset_ptr,d_scc_count,V,i_status);

        thrust::exclusive_scan(scc_offset.begin(),scc_offset.end(),scc_offset.begin());

        int h_scc_count; cudaMemcpy(&h_scc_count,d_scc_count,sizeof(int),cudaMemcpyDeviceToHost);

        set_zero_nodes_size(current,h_scc_count);

        int*new_nodes_count; cudaMalloc(&new_nodes_count,sizeof(int)); cudaMemset(new_nodes_count,0,sizeof(int));

        int blocks = (h_scc_count + BLOCKSIZE - 1) / BLOCKSIZE;

        d_scc_ptr = thrust::raw_pointer_cast(scc.data()); d_scc_offset_ptr = thrust::raw_pointer_cast(scc_offset.data());
        assign_single_nodes<<<blocks,BLOCKSIZE>>>(current,d_scc_ptr,d_scc_offset_ptr,d_scc_count,new_nodes_count,i_status,V);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error assign: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        int h_new_nodes; cudaMemcpy(&h_new_nodes,new_nodes_count,sizeof(int),cudaMemcpyDeviceToHost);

        if(h_new_nodes > 0)
        {

          int total_nodes = G->map.size();
          int h_map_index; cudaMemcpyFromSymbol(&h_map_index,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);

          if(h_map_index + h_new_nodes >= total_nodes)
          {
            G->map.resize(total_nodes+h_new_nodes);
          }

          node*nodes_ptr = thrust::raw_pointer_cast(G->map.data());

          thrust::host_vector<int>h_scc_offset = scc_offset;
          thrust::host_vector<int>sccindex(h_new_nodes);
          thrust::host_vector<node*>wanted_nodes;

          int count = 0;
          for(int i=0;i<h_scc_count;i++)
          {
            int scc_size = h_scc_offset[i+1] - h_scc_offset[i];
            if(scc_size > 1)
            {
              if(scc_size != h_nodes_size || i_status)
              {
                int*temp; cudaMalloc(&temp,scc_size*sizeof(int));
                cudaMemcpy(&((nodes_ptr+h_map_index+count)->nodes),&temp,sizeof(int*),cudaMemcpyHostToDevice);
                wanted_nodes.push_back(nodes_ptr+h_map_index+count); q.push(nodes_ptr+h_map_index+count); 
                sccindex[count] = i;
                count++;
              }
            }
          }

          thrust::device_vector<int>d_scc_index = sccindex;
          int*scc_index = thrust::raw_pointer_cast(d_scc_index.data()); nodes_ptr = thrust::raw_pointer_cast(G->map.data());
          d_scc_ptr = thrust::raw_pointer_cast(scc.data()); d_scc_offset_ptr = thrust::raw_pointer_cast(scc_offset.data());

          blocks = (h_new_nodes + BLOCKSIZE - 1) / BLOCKSIZE;
          fill_new_nodes<<<blocks,BLOCKSIZE>>>(scc_index,current,d_scc_ptr,d_scc_offset_ptr,new_nodes_count,i_status,nodes_ptr); cudaDeviceSynchronize();
          err = cudaGetLastError();
          if (err != cudaSuccess) {
              std::cerr << "CUDA Error fill new nodes : " << cudaGetErrorString(err) << std::endl;
              return;
          }

          h_map_index = h_map_index + h_new_nodes;
          cudaMemcpyToSymbol(::map_index,&h_map_index,sizeof(int),0,cudaMemcpyHostToDevice);

          nodes_ptr = thrust::raw_pointer_cast(G->map.data()); int h_label; int*d_label; cudaMalloc(&d_label,sizeof(int));
          //Can use threads here
          for(int i=0;i<h_new_nodes;i++)
          {
            set_nn_nodes_parent(wanted_nodes[i],nodes_ptr);

            get_label<<<1,1>>>(wanted_nodes[i],d_label); cudaDeviceSynchronize();
            cudaMemcpy(&h_label,d_label,sizeof(int),cudaMemcpyDeviceToHost);

            if(h_label < 0)
            {
              change_node_subtree(h_label,nodes_ptr);
            }
          }

          //Distribution of edges based on the new nodes
          if(h_new_nodes > 0)
          {
            get_nodes_size<<<1,1>>>(current,d_nodes_size); cudaDeviceSynchronize(); 
            cudaMemcpy(&h_nodes_size,d_nodes_size,sizeof(int),cudaMemcpyDeviceToHost);
            node**wanted_nodes_ptr = thrust::raw_pointer_cast(wanted_nodes.data());
            edges = thrust::raw_pointer_cast(G->edges.data()); edgesize = G->edges.size();
            distribute_Edges(current,h_nodes_size,wanted_nodes_ptr,h_new_nodes,edges,edgesize,V);
          }
        }
        i_status = false;
      }
    }
}

__global__ void create_edge(edge*e,int label,int f_vertex,int s_vertex,int h_connect1,int h_connect2)
{
  e->label = label;
  e->vertices[0] = f_vertex;
  e->vertices[1] = s_vertex;
  e->connect[0] = h_connect1;
  e->connect[1] = h_connect2;
}

__global__ void copy_edges(edge**new_edges,edge**edges,int edgesize,edge*e)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < edgesize)
  {
    new_edges[tid] = edges[tid];
  }
  if(tid == 0)
  {
    new_edges[edgesize] = e;
  }
}

void Add_Edge(int u,int v,SCC_tree*G)
{
    node*nodes_ptr = thrust::raw_pointer_cast(G->map.data());
    int mapsize; cudaMemcpyFromSymbol(&mapsize,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);

    //Adding the edge to the graph
    int blocks = (mapsize + BLOCKSIZE - 1) / BLOCKSIZE;

    //Finding the Least Common Ancestor
    node**address1; cudaMalloc(&address1,sizeof(node*));
    node**address2; cudaMalloc(&address2,sizeof(node*));

    find_address<<<blocks,BLOCKSIZE>>>(nodes_ptr,mapsize,u,v,address1,address2);
    cudaDeviceSynchronize();

    int*f_vertex; int*s_vertex; bool*d_status; int*d_label; int*d_iv;
    cudaMalloc(&f_vertex,sizeof(int)); 
    cudaMalloc(&s_vertex,sizeof(int));
    cudaMalloc(&d_status,sizeof(bool));
    cudaMalloc(&d_label,sizeof(int));
    cudaMalloc(&d_iv,sizeof(int));

    node**LCA; cudaMalloc(&LCA,sizeof(node*));
    findLCA<<<1,1>>>(address1,address2,LCA,f_vertex,s_vertex,d_status,d_label,d_iv,G->master_node);
    cudaDeviceSynchronize();

    int h_f_vertex; int h_s_vertex; int h_iv; int h_label; bool h_status;
    cudaMemcpy(&h_f_vertex,f_vertex,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_s_vertex,s_vertex,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_iv,d_iv,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_label,d_label,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_status,d_status,sizeof(bool),cudaMemcpyDeviceToHost);


    bool induced = false;
    int h_connect1 = 0; int h_connect2 = 0;
    if(h_f_vertex>=0 && h_s_vertex>=0)
    {
      if(h_s_vertex == h_iv && !h_status){h_s_vertex = h_iv+G->n_v; induced = true;}
    }
    else if(h_f_vertex>=0 && h_s_vertex<0)
    {
      if(h_s_vertex == h_iv && !h_status){h_s_vertex = (-1)*h_iv+G->n_v; induced = true;}
      h_connect2 = v; 
    }
    else if(h_f_vertex<0 && h_s_vertex>=0)
    {
      if(h_s_vertex == h_iv && !h_status){h_s_vertex = h_iv+G->n_v; induced = true;}
      h_connect1 = u;
    }
    else if(h_f_vertex<0 && h_s_vertex<0)
    {
      if(h_s_vertex == h_iv && !h_status){h_s_vertex = (-1)*h_iv+G->n_v; induced = true;}
      h_connect1 = u; h_connect2 = v;
    }

    edge*e; cudaMalloc(&e,sizeof(edge));
    create_edge<<<1,1>>>(e,h_label,h_f_vertex,h_s_vertex,h_connect1,h_connect2);
    cudaDeviceSynchronize();

    int edgesize = G->edges.size(); edge**edges = thrust::raw_pointer_cast(G->edges.data());
    edge**new_edges; cudaMalloc(&new_edges,sizeof(edge*)*(edgesize+1));

    blocks = (edgesize + BLOCKSIZE -1 )/BLOCKSIZE;
    copy_edges<<<blocks,BLOCKSIZE>>>(new_edges,edges,edgesize,e); cudaDeviceSynchronize();

    G->edges.resize(edgesize+1); 

    edges = thrust::raw_pointer_cast(G->edges.data());
    cudaMemcpy(edges,new_edges,sizeof(edge*)*(edgesize+1),cudaMemcpyDeviceToDevice);

    if(!induced)
    {
      incremental(LCA,G,h_status);
    }
}

#endif // SCC_INCREMENTAL_H