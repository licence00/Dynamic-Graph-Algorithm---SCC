#include <fstream>
#include <string>
#include <random>
#include <vector>
#include <iostream>
#include <set>
#include "load.h"
using namespace std;

void load_Graph(string filename,int*Vertices,int*Edges,int*Updates,int**edges,vector<vector<int>>&updates,float percentage)
{
    int V,E,U,type;
    int firstVertex, secondVertex;

    // open the file
    filebuf fb;
    fb.open(filename,ios::in);
    if (!fb.is_open() )
    {
      printf("Error Reading graph file\n");
      return;
    }
    istream is(&fb);

    //obtain the size of the graph (Edges, Vertices)
    is >> V >> E;
    printf("V: %d, E: %d\n", V, E);

    //Randomly pick edges that needed to be updateed
    //***************************In Aqua *********************************

    // Allocate memory for edges
    *edges = new int[E*2];
    for(int i=0;i<E;i++)
    {
        is>>firstVertex>>secondVertex;
        (*edges)[i*2] = firstVertex;
        (*edges)[i*2+1] = secondVertex;
    }

    // std::ofstream outfile("edges.txt");
    // if (!outfile.is_open()) {
    //     std::cerr << "Error opening file for writing." << std::endl;
    //     exit(0);
    // }

    float temp = (percentage * E * 0.01);
    U = (int)temp;

    type = 0; int delete_count = U/2;
    std::random_device rd1; // Non-deterministic generator
    std::mt19937 gen1(rd1()); // Mersenne Twister generator
    std::uniform_int_distribution<> distrib1(0, E-1);

    std::set<int>random_values;
    for (int i = 0; i < delete_count; ++i) {
        int number = distrib1(gen1);
        random_values.insert(number);
    }
    
    delete_count = random_values.size();
    
    for (auto it = random_values.begin(); it != random_values.end(); ++it) {
        int value = *it;
        updates.push_back({type, (*edges)[value*2], (*edges)[value*2+1]});
    }

    type = 1; int add_count = U - delete_count;

    std::random_device rd2; // Non-deterministic generator
    std::mt19937 gen2(rd2()); // Mersenne Twister generator
    std::uniform_int_distribution<> distrib2(0, V-1);

    std::vector<std::pair<int,int>>add_edges;
    for (int i = 0; i < add_count; ++i) {
        int number1 = distrib2(gen2);
        int number2 = distrib2(gen2);
        if(number1 != number2)
        {
            add_edges.push_back({number1,number2});
        }
    }
    
    add_count = add_edges.size();
    
    for(int i=0;i<add_count;i++)
    {
        updates.push_back({type, add_edges[i].first, add_edges[i].second});
    }

    U = delete_count + add_count;
    printf("U: %d\n", U);

    // outfile.close();

    // is >> U;
    // *updates = new int*[U];
    // for(int i=0;i<U;i++)
    // {
    //     (*updates)[i] = new int[3];
    //     is>>type>>firstVertex>>secondVertex;
    //     (*updates)[i][0] = type;
    //     (*updates)[i][1] = firstVertex;
    //     (*updates)[i][2] = secondVertex;
    // }

    fb.close();


    *Vertices = V; *Edges = E; *Updates = U;
}
