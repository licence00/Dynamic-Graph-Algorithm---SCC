#ifndef SCC_Tree_H
#define SCC_Tree_H

#include <queue>
#include <chrono>
#include <thread>
#include "SCC_Map.h"
using namespace std;

__device__ int map_index;
__device__ int label_count;

class SCC_tree
{
    public:
    int n_v; int n_e;
    thrust::device_vector<edge*>edges;
    thrust::device_vector<node>map;
    node*master_node;
    SCC_tree(int vertices,int edgesize)
    {
        this->n_v = vertices; this->n_e = edgesize;
        cudaMalloc(&master_node,sizeof(node));
        map = thrust::device_vector<node>(2*vertices);
        // map = thrust::device_vector<node>(vertices);
        edges = thrust::device_vector<edge*>(edgesize);
    }
};

typedef struct
{
  node*main;
  SCC_tree*tree;
} scc_args;

void set_nodes_size(node*org,int size)
{
    int*temp; cudaMalloc(&temp,sizeof(int)*size);
    cudaMemcpy(&(org->nodes),&temp,sizeof(int*),cudaMemcpyHostToDevice);
    cudaMemcpy(&(org->nodes_size),&size,sizeof(int),cudaMemcpyHostToDevice);
}

void set_zero_nodes_size(node*org,int size)
{
    int*temp; cudaMalloc(&temp,sizeof(int)*size);
    cudaMemcpy(&(org->nodes),&temp,sizeof(int*),cudaMemcpyHostToDevice);
    int zero = 0; cudaMemcpy(&(org->nodes_size),&zero,sizeof(int),cudaMemcpyHostToDevice);
}

__global__ void set_master_node(node*org)
{
    org->nodes = nullptr;
    org->nodes_size = 0;
    org->label = -1;
    org->parent = nullptr;
    org->depth = 0;
    org->primary_ancestor = -1;
    org->induce_vertex = 0;
}

__global__ void set_induce_vertex(node*org,bool*val)
{
    if(!(*val))
    {
        org->induce_vertex = org->nodes[0];
    }
}

__global__ void get_nodes_size(node*org,int*val)
{
    *val = org->nodes_size;
}

template <typename T>
__global__ void find_node(node*current,int size, T value, bool*val)
{
    unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < size) 
    {
        if (current->nodes[id] == value) 
        {
            *val = true;
        }
    }
}

__global__ void distribute_nodes(bool status,node*new_nodes,int*scc,int*scc_offset,int*scc_count,node*p,int V)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < *scc_count)
    {
        int scc_size = scc_offset[tid+1]-scc_offset[tid];
        if(scc_size == 1)
        {
            new_nodes[map_index+tid].nodes[0] = scc[scc_offset[tid]];
            new_nodes[map_index+tid].label = new_nodes[map_index+tid].nodes[0];
        }
        else if(scc_size > 1)
        {
            for(int index=0;index<scc_size;index++)
            {
                new_nodes[map_index+tid].nodes[index] = scc[scc_offset[tid]+index];
            }
            new_nodes[map_index+tid].label = (-1)*(atomicAdd(&label_count,1));
        }
        new_nodes[map_index+tid].nodes_size = scc_size;
        new_nodes[map_index+tid].induce_vertex = new_nodes[map_index+tid].nodes[0];
        if(status) //at master node case
        {
            new_nodes[map_index+tid].parent = nullptr;
            new_nodes[map_index+tid].depth = 0;
            new_nodes[map_index+tid].primary_ancestor = new_nodes[map_index+tid].label; 
        }
        else
        {
            new_nodes[map_index+tid].parent = p;
            new_nodes[map_index+tid].depth = p->depth+1;
            new_nodes[map_index+tid].primary_ancestor = p->primary_ancestor; 
        }
        p->nodes[tid] = new_nodes[map_index+tid].label;
    }
}

__global__ void distribute_single_nodes(node*new_nodes,int nodes_size,node*parent)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < nodes_size)
    {
        int vertex = parent->nodes[tid];
        new_nodes[map_index+tid].nodes[0] = vertex;
        new_nodes[map_index+tid].label = vertex;
        new_nodes[map_index+tid].nodes_size = 1;
        new_nodes[map_index+tid].induce_vertex = vertex;
        new_nodes[map_index+tid].parent = parent;
        new_nodes[map_index+tid].depth = parent->depth+1;
        new_nodes[map_index+tid].primary_ancestor = parent->primary_ancestor;
    }
}

__global__ void master_edges(edge**edges,int edgesize,node*current,node*new_node,int V)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < edgesize)
    {
        if(edges[tid]->label == current->label)
        {
            int f_vertex = edges[tid]->vertices[0]; int s_vertex = edges[tid]->vertices[1];
            bool bool1 = false; bool bool2 = false;
            for(int i=0;i<new_node->nodes_size;i++)
            {
                if(new_node->nodes[i] == f_vertex)
                {
                    bool1 = true;
                }
                if(new_node->nodes[i] == s_vertex)
                {
                    bool2 = true;
                }
            }
            if(bool1 && bool2)
            {
                if(s_vertex == new_node->induce_vertex)
                {
                    if(new_node->induce_vertex < 0){s_vertex = (-1)*(new_node->induce_vertex)+V;}
                    else{s_vertex = (new_node->induce_vertex)+V;}
                }
                edges[tid]->label = new_node->label;
                edges[tid]->vertices[0] = f_vertex; edges[tid]->vertices[1] = s_vertex;
            }
            else if(bool1)
            {
                if(f_vertex>=0)
                {
                    edges[tid]->vertices[0] = new_node->label;
                    edges[tid]->connect[0] = f_vertex;
                }
                else
                {
                    edges[tid]->vertices[0] = new_node->label;
                }
            }
            else if(bool2)
            {
                if(s_vertex>=0)
                {
                    edges[tid]->vertices[1] = new_node->label;
                    edges[tid]->connect[1] = s_vertex;
                }
                else
                {
                    edges[tid]->vertices[1] = new_node->label;
                }
            }
        }
    }
}

__global__ void processedges(edge**edges,int edgesize,node*current,node*new_node,int V,int old_ivertex)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < edgesize)
    {
        if(edges[tid]->label == current->label)
        {
            int f_vertex = edges[tid]->vertices[0]; int s_vertex = edges[tid]->vertices[1];
            if(old_ivertex < 0 && s_vertex == -1*(old_ivertex)+V){s_vertex = old_ivertex;}
            else if(old_ivertex >= 0 && s_vertex == old_ivertex+V){s_vertex = old_ivertex;}
            bool bool1 = false; bool bool2 = false;
            for(int i=0;i<new_node->nodes_size;i++)
            {
                if(new_node->nodes[i] == f_vertex)
                {
                    bool1 = true;
                }
                if(new_node->nodes[i] == s_vertex)
                {
                    bool2 = true;
                }
            }
            if(bool1 && bool2)
            {
                if(s_vertex == new_node->induce_vertex)
                {
                    if(new_node->induce_vertex < 0){s_vertex = (-1)*(new_node->induce_vertex)+V;}
                    else{s_vertex = (new_node->induce_vertex)+V;}
                }
                edges[tid]->label = new_node->label;
                edges[tid]->vertices[0] = f_vertex; edges[tid]->vertices[1] = s_vertex;
            }
            else if(bool1)
            {
                if(f_vertex>=0)
                {
                    edges[tid]->vertices[0] = new_node->label;
                    edges[tid]->connect[0] = f_vertex;
                }
                else
                {
                    edges[tid]->vertices[0] = new_node->label;
                }
            }
            else if(bool2)
            {
                int org = s_vertex;
                s_vertex = new_node->label;
                if(s_vertex == current->induce_vertex)
                {
                    if(current->induce_vertex < 0){s_vertex = (-1)*(current->induce_vertex)+V;}
                    else{s_vertex = (current->induce_vertex)+V;}
                }
                if(org>=0)
                {
                    edges[tid]->vertices[1] = s_vertex;
                    edges[tid]->connect[1] = org;
                }
                else
                {
                    edges[tid]->vertices[1] = s_vertex;
                }
            }
        }
    }
}

void distribute_Edges(node*current,int nodesize,node**new_nodes,int new_nodes_size,edge**edges,int total_edges,int V)
{

    int*d_old_ivertex; cudaMalloc(&d_old_ivertex,sizeof(int));
    get_induce_vertex<<<1,1>>>(current,d_old_ivertex); cudaDeviceSynchronize();
    int old_iv; cudaMemcpy(&old_iv,d_old_ivertex,sizeof(int),cudaMemcpyDeviceToHost);
    
    bool*bool1; cudaMalloc(&bool1,sizeof(bool)); cudaMemset(bool1,false,sizeof(bool));

    /*Checking whether the current node contains the initial induced vertex as its child or not*/
    int blocks1 = (nodesize + BLOCKSIZE - 1) / BLOCKSIZE;
    find_node<int><<<blocks1,BLOCKSIZE>>>(current,nodesize,old_iv,bool1);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in distribute : " << cudaGetErrorString(err) << std::endl;
        return;
    }

    /*If not assign the current induce vertex with a new one*/    
    set_induce_vertex<<<1,1>>>(current,bool1);
    cudaDeviceSynchronize();

    int blocks = (total_edges + BLOCKSIZE - 1) / BLOCKSIZE;

    for(int i=0;i<new_nodes_size;i++)
    {
        processedges<<<blocks,BLOCKSIZE>>>(edges,total_edges,current,new_nodes[i],V,old_iv);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error process : " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }
}

void finish_tree(node*current,SCC_tree*G)
{
    int h_nodes_size; int*d_nodes_size; cudaMalloc(&d_nodes_size,sizeof(int));
    get_nodes_size<<<1,1>>>(current,d_nodes_size); cudaDeviceSynchronize();
    cudaMemcpy(&h_nodes_size,d_nodes_size,sizeof(int),cudaMemcpyDeviceToHost);

    node*nodes_ptr = thrust::raw_pointer_cast(G->map.data());

    int h_map_index; cudaMemcpyFromSymbol(&h_map_index,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);

    for(int i=0;i<h_nodes_size;i++)
    {
        int scc_size = 1;
        int*temp; cudaMalloc(&temp,sizeof(int)*scc_size);
        cudaMemcpy(&((nodes_ptr+h_map_index+i)->nodes),&temp,sizeof(int*),cudaMemcpyHostToDevice);
    }

    int blocks = (h_nodes_size + BLOCKSIZE - 1) / BLOCKSIZE;
    nodes_ptr = thrust::raw_pointer_cast(G->map.data());
    distribute_single_nodes<<<blocks,BLOCKSIZE>>>(nodes_ptr,h_nodes_size,current);
    cudaDeviceSynchronize();

    h_map_index = h_map_index+h_nodes_size;
    cudaMemcpyToSymbol(::map_index,&h_map_index, sizeof(int) , 0 , cudaMemcpyHostToDevice);

}

__global__ void depth_of_node(node*org,int*val)
{
    *val = org->depth;
}

void find_internal_structure(node*main,SCC_tree*G)
{
    int edgesize = G->edges.size(); int V = G->n_v;
    
    queue<node*>q; q.push(main);
    
    while(!q.empty())
    {
        node*current = q.front(); q.pop();

        int depth; int*d_depth; cudaMalloc(&d_depth,sizeof(int));
        depth_of_node<<<1,1>>>(current,d_depth); cudaDeviceSynchronize();
        cudaMemcpy(&depth,d_depth,sizeof(int),cudaMemcpyDeviceToHost);

        if(depth>5)
        {
            finish_tree(current,G); //If the depth of the node is greater than 5 then finish the tree
        }
        else
        {

            int*d_nodes_size; cudaMalloc(&d_nodes_size,sizeof(int));
            get_nodes_size<<<1,1>>>(current,d_nodes_size); cudaDeviceSynchronize();
            int h_nodes_size; cudaMemcpy(&h_nodes_size,d_nodes_size,sizeof(int),cudaMemcpyDeviceToHost);

            thrust::device_vector<int>scc(h_nodes_size,0);
            thrust::device_vector<int>scc_offset(h_nodes_size+1,0);

            int*d_scc_ptr = thrust::raw_pointer_cast(scc.data());
            int*d_scc_offset_ptr = thrust::raw_pointer_cast(scc_offset.data());
            int h_scc_count = 0; int*d_scc_count; cudaMalloc(&d_scc_count,sizeof(int));
            cudaMemcpy(d_scc_count,&h_scc_count,sizeof(int),cudaMemcpyHostToDevice);

            /*Apply SCC algorithm in order to create the current node children*/
            bool status = false; edge**edges = thrust::raw_pointer_cast(G->edges.data());
            SCC_With_Mapping(edges,edgesize,current,h_nodes_size,d_scc_ptr,d_scc_offset_ptr,d_scc_count,V,status);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error scc map : " << cudaGetErrorString(err) << std::endl;
                return;
            }

            thrust::exclusive_scan(scc_offset.begin(),scc_offset.end(),scc_offset.begin());

            cudaMemcpy(&h_scc_count,d_scc_count,sizeof(int),cudaMemcpyDeviceToHost);

            /**************************************************/

            thrust::host_vector<int>h_scc_offset = scc_offset;

            thrust::host_vector<node*>wanted_nodes;

            node*nodes_ptr = thrust::raw_pointer_cast(G->map.data());

            int h_map_index; cudaMemcpyFromSymbol(&h_map_index,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);

            for(int i=0;i<h_scc_count;i++)
            {
                int scc_size = h_scc_offset[i+1]-h_scc_offset[i];
                int*temp; cudaMalloc(&temp,sizeof(int)*scc_size);
                cudaMemcpy(&((nodes_ptr+h_map_index+i)->nodes),&temp,sizeof(int*),cudaMemcpyHostToDevice);
                if(scc_size>1)
                {
                    wanted_nodes.push_back(nodes_ptr+h_map_index+i); 
                    q.push(nodes_ptr+h_map_index+i);
                }
            }

            set_nodes_size(current,h_scc_count);
            
            int blocks = (h_scc_count + BLOCKSIZE - 1) / BLOCKSIZE;

            //Creating the new nodes formed and assign the corresponding nodes information
            int status_n = false; nodes_ptr = thrust::raw_pointer_cast(G->map.data());
            d_scc_ptr = thrust::raw_pointer_cast(scc.data()); d_scc_offset_ptr = thrust::raw_pointer_cast(scc_offset.data());
            distribute_nodes<<<blocks,BLOCKSIZE>>>(status_n,nodes_ptr,d_scc_ptr,d_scc_offset_ptr,d_scc_count,current,V);
            cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error distrubute 1 : " << cudaGetErrorString(err) << std::endl;
                return;
            }

            h_map_index = h_map_index+h_scc_count;
            cudaMemcpyToSymbol(::map_index,&h_map_index, sizeof(int) , 0 , cudaMemcpyHostToDevice);

            int wanted_nodes_size = wanted_nodes.size();

            //Distributing the edges based on the SCCs formed
            if(wanted_nodes_size>0)
            {                
                node**wanted_nodes_ptr = thrust::raw_pointer_cast(wanted_nodes.data());
                distribute_Edges(current,h_scc_count,wanted_nodes_ptr,wanted_nodes_size,edges,edgesize,V);
            }
            
            h_scc_offset.clear(); scc.clear(); scc_offset.clear(); wanted_nodes.clear(); 
        }
    } 
}

void create_tree(SCC_tree*T,FILE*file)
{
    int V = T->n_v; int E = T->n_e;

    //Creating master node
    cudaMalloc(&(T->master_node),sizeof(node));
    set_master_node<<<1,1>>>(T->master_node); cudaDeviceSynchronize();

    int h_map_index = 0;
    cudaMemcpyToSymbol(::map_index,&h_map_index, sizeof(int) , 0 , cudaMemcpyHostToDevice);

    /*Starting it from 2 because the master node takes the value -1 as its label*/
    int h_label_count = 2;
    cudaMemcpyToSymbol(::label_count,&h_label_count, sizeof(int) , 0 , cudaMemcpyHostToDevice);

    int h_scc_count; int*d_scc_count; cudaMalloc(&d_scc_count,sizeof(int));
    thrust::device_vector<int>d_scc(V,0);
    thrust::device_vector<int>d_scc_offset(V+1,0);

    int*d_scc_ptr = thrust::raw_pointer_cast(d_scc.data());
    int*d_scc_offset_ptr = thrust::raw_pointer_cast(d_scc_offset.data());

    edge**edges = thrust::raw_pointer_cast(T->edges.data());

    csr_format*csr = new csr_format(V,E);  int master_label = -1;

    csr->create_csr_format(edges,E,master_label);
    
    bool status = true;
    SCC(csr,d_scc_ptr,d_scc_offset_ptr,d_scc_count,status);

    thrust::exclusive_scan(d_scc_offset.begin(),d_scc_offset.end(),d_scc_offset.begin());

    cudaMemcpy(&h_scc_count,d_scc_count,sizeof(int),cudaMemcpyDeviceToHost);

    /**************************************************/

    thrust::host_vector<int>h_scc_offset = d_scc_offset;

    thrust::host_vector<node*>wanted_nodes;

    node*nodes_ptr = thrust::raw_pointer_cast(T->map.data());

    cudaMemcpyFromSymbol(&h_map_index,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);

    
    for(int i=0;i<h_scc_count;i++)
    {
        int scc_size = h_scc_offset[i+1]-h_scc_offset[i];
        int*temp; cudaMalloc(&temp,sizeof(int)*scc_size);
        cudaMemcpy(&((nodes_ptr+h_map_index+i)->nodes),&temp,sizeof(int*),cudaMemcpyHostToDevice);
        if(scc_size>1)
        {
          wanted_nodes.push_back(nodes_ptr+h_map_index+i);
        }
    }

    set_nodes_size(T->master_node,h_scc_count);

    int blocks = (h_scc_count + BLOCKSIZE - 1) / BLOCKSIZE;
    nodes_ptr = thrust::raw_pointer_cast(T->map.data()); status = true;
    d_scc_ptr = thrust::raw_pointer_cast(d_scc.data()); d_scc_offset_ptr = thrust::raw_pointer_cast(d_scc_offset.data());
    distribute_nodes<<<blocks,BLOCKSIZE>>>(status,nodes_ptr,d_scc_ptr,d_scc_offset_ptr,d_scc_count,T->master_node,V);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error distribute : " << cudaGetErrorString(err) << std::endl;
        return;
    }

    h_map_index = h_map_index+h_scc_count;
    cudaMemcpyToSymbol(::map_index,&h_map_index, sizeof(int) , 0 , cudaMemcpyHostToDevice);

    int wanted_nodes_size = wanted_nodes.size();
   
    if(wanted_nodes_size > 0)
    {
        int edgesize = T->edges.size();
        edge**edges = thrust::raw_pointer_cast(T->edges.data());
    
        int blocks = (edgesize + BLOCKSIZE - 1) / BLOCKSIZE;
        for(int i=0;i<wanted_nodes_size;i++)
        {
            master_edges<<<blocks, BLOCKSIZE>>>(edges,edgesize,T->master_node,wanted_nodes[i],V);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error process master : " << cudaGetErrorString(err) << std::endl;
                return;
            }
        }
    }
    d_scc.clear(); d_scc_offset.clear(); h_scc_offset.clear();

    for(int i=0;i<wanted_nodes_size;i++)
    {
        find_internal_structure(wanted_nodes[i],T);
    }

    cudaMemcpyFromSymbol(&h_map_index,::map_index,sizeof(int),0,cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&h_label_count,::label_count,sizeof(int),0,cudaMemcpyDeviceToHost);
    printf("Map Index = %d\n",h_map_index);
    printf("Label Count = %d\n",h_label_count);
}

__global__ void get_scc_count(node*master,int*val)
{
    *val = master->nodes_size;
}

#endif