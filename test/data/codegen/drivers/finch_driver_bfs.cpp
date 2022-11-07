//
// Created by Alexandra Dima on 30.10.2022.
//

#include "../../../tmp/codegen/bfs.h"
#include <string>
#include <fstream>
#include "julia.h"
#include "einsum_taco/gstrt/runtime.h"

int N;
int source;
jl_value_t* edges;

int main(int argc, char* argv[]) {
    Graph g = Graph{};
    N = make_weights_and_edges(argv[1], &g);

    edges = g.edges;
    source = atoi(argv[2]);

    Main();
}