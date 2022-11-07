//
// Created by Alexandra Dima on 30.10.2022.
//

#include "../../../tmp/codegen/pagerank.h"
#include <string>
#include <fstream>
#include "julia.h"
#include "einsum_taco/gstrt/runtime.h"

int N;
int source;
float damp;
float beta_score;
jl_value_t* edges;

int main(int argc, char* argv[]) {
    Graph g = Graph{};
    N = make_weights_and_edges(argv[1], &g);

    edges = g.edges;
    source = atoi(argv[2]);
    damp = atof(argv[2]); //0.85
    beta_score = (1.0 - damp) / N;

    Main();
}