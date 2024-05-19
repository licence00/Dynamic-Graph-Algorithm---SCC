#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include "SCC.h"
#include "SCC_Map.h"
#include "SCC_Tree.h"
#include "SCC_Incremental.h"
#include "load.h"


using namespace std;

#define BLOCKSIZE 1024

// __global__ void print_scc(edge**edges,int edgesize,node*master_node,node*label)
// {
//     printf("Printing\n\n");
//     printf("mASTER NODE:\n");
//     printf("Label: %d\n",master_node->label);
//     printf("Induce Vertex: %d\n",master_node->induce_vertex);
//     printf("Depth: %d\n",master_node->depth);
//     printf("Primary Ancestor: %d\n",master_node->primary_ancestor);
//     printf("Nodes are : \n");
//     for(int i=0;i<master_node->nodes_size;i++)
//     {
//         printf("%d ",master_node->nodes[i]);
//     }
//     printf("\n\n");
//     printf("Edges are of size  : %d\n",edgesize);
//     for(int i=0;i<edgesize;i++)
//     {
//         edge*e = edges[i];
//         printf("Edge %d\n",i);
//         printf("Label: %d\n",e->label);
//         printf("Vertices: %d %d\n",e->vertices[0],e->vertices[1]);
//         printf("Connect: %d %d\n",e->connect[0],e->connect[1]);
//     }
//     printf("\n\n");
//     for(int j=0;j<::map_index;j++)
//     {
//         printf("Label: %d\n",label[j].label);
//         if(label[j].label<0)
//         {
//           if(label[j].parent != nullptr)
//           {
//             printf("Parent: %d\n",label[j].parent->label);
//           }
//           else
//           {
//             printf("Hey I am root\n");
//           }
//           printf("Induce Vertex: %d\n",label[j].induce_vertex);
//           printf("Depth: %d\n",label[j].depth);
//           printf("Primary Ancestor: %d\n",label[j].primary_ancestor);
//           printf("Nodes size is %d\n",label[j].nodes_size);
//           printf("Nodes are : \n");
//           for(int i=0;i<label[j].nodes_size;i++)
//           {
//               printf("%d ",label[j].nodes[i]);
//           }
//           printf("\n\n");
//         }
//         else
//         {
//           printf("Depth: %d\n",label[j].depth);
//           printf("Primary Ancestor: %d\n",label[j].primary_ancestor);
//           printf("Nodes size is %d\n",label[j].nodes_size);
//           printf("\n\n");
//         }
//     }
//     printf("\n");
// }

__global__ void create_edges(edge**edges,int*other_edges,int E)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index<E)
    {
        edges[index]->label = -1;
        edges[index]->vertices[0] = other_edges[2*index];
        edges[index]->vertices[1] = other_edges[2*index+1];
        edges[index]->connect[0] = 0; edges[index]->connect[1] = 0;
    }
}

int main(int argc, char* argv[]) 
{
    int V,E,U;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <float_value>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    // Convert the second argument to a float
    float percentage;
    try {
        percentage = std::stof(argv[2]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << argv[2] << " is not a valid float" << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cerr << "Argument out of range: " << argv[2] << " is out of range for a float" << std::endl;
        return 1;
    }

    string output = "Result.txt";
    FILE *file = fopen(output.c_str(), "w");

    int*edges = NULL; vector<vector<int>>updates;

    load_Graph(filename,&V,&E,&U,&edges,updates,percentage);

    SCC_tree*tree = new SCC_tree(V,E);

    edge*d_org_edges; cudaMalloc(&d_org_edges,E*sizeof(edge));

    for(int i=0;i<E;i++)
    {
        tree->edges[i] = d_org_edges+i;
    }

    int*d_edges; cudaMalloc(&d_edges,2*E*sizeof(int));
    cudaMemcpy(d_edges,edges,2*E*sizeof(int),cudaMemcpyHostToDevice);
    free(edges);
    
    edge**edges_ptr = thrust::raw_pointer_cast(tree->edges.data());

    int blocks = (E+BLOCKSIZE-1)/BLOCKSIZE;
    create_edges<<<blocks,BLOCKSIZE>>>(edges_ptr,d_edges,E); cudaDeviceSynchronize();
    cudaFree(d_edges);
    
    // fprintf(file,"edges copied\n");

    cudaEvent_t start,stop; float elapsedtime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    create_tree(tree,file); // Create the tree

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedtime, start, stop);

    fprintf(file,"Time taken for the tree creation is %f seconds\n",elapsedtime/1000);

    int scc_count; int*d_scc_count; cudaMalloc(&d_scc_count,sizeof(int));
    get_scc_count<<<1,1>>>(tree->master_node,d_scc_count); cudaDeviceSynchronize();
    cudaMemcpy(&scc_count,d_scc_count,sizeof(int),cudaMemcpyDeviceToHost);
    fprintf(file,"The number of SCCs in the tree is %d\n",scc_count);

    cudaEvent_t start_d,stop_d;
    float elapsedtime_d;

    // Create events
    cudaEventCreate(&start_d);
    cudaEventCreate(&stop_d);

    cudaEventRecord(start_d, 0);

    for (int i = 0; i < U; ++i) 
    {
        if(updates[i][0] == 0)
        {
            if(updates[i][1] != updates[i][2])
            {
                Remove_Edge(updates[i][1],updates[i][2],tree);
            }
        }
        else if(updates[i][0] == 1)
        {
            if(updates[i][1] != updates[i][2])
            {
                Add_Edge(updates[i][1],updates[i][2],tree);
            }
        }
    }

    cudaEventRecord(stop_d, 0);

    cudaEventSynchronize(stop_d);

    cudaEventElapsedTime(&elapsedtime_d, start_d, stop_d);

    fprintf(file,"Time taken for the update is %f seconds\n",elapsedtime_d/1000);

    get_nodes_size<<<1,1>>>(tree->master_node,d_scc_count); cudaDeviceSynchronize();
    cudaMemcpy(&scc_count,d_scc_count,sizeof(int),cudaMemcpyDeviceToHost);

    fprintf(file,"SCC Count after update is %d\n",scc_count);

    // edges_ptr = thrust::raw_pointer_cast(tree->edges.data());
    // int edgesize = tree->edges.size(); node*label = thrust::raw_pointer_cast(tree->map.data());
    // print_scc<<<1,1>>>(edges_ptr,edgesize,tree->master_node,label); cudaDeviceSynchronize();

    // filebuf fb1;
    // fb1.open(filename,ios::in);
    // if (!fb1.is_open() )
    // {
    //   printf("Error Reading graph file\n");
    //   return;
    // }
    // istream is1(&fb1);

    // for(int i=0;i<E+U+2;i++)
    // {
    //     string s; getline(is1,s);
    // }
    // int Q;
    // is1 >> Q; int first,second;
    // printf("Q is %d\n",Q);
    // for(int i=0;i<Q;i++)
    // {
    //     is1 >> first >> second;
    //     int ans; Query(first,second,tree,&ans);
    //     if(ans == 1)
    //     {
    //         printf("YES\n");
    //     }
    //     else
    //     {
    //         printf("NO\n");
    //     }
    // }
    // fb1.close();

    return 0;
}
