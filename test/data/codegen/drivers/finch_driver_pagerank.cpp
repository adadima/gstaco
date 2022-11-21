//
// Created by Alexandra Dima on 30.10.2022.
//

#include "../../../tmp/codegen/pagerank.h"
#include <string>
#include <fstream>
#include "julia.h"
#include <iostream>

int N;
float damp;
float beta_score;
jl_value_t* edges;

int main(int argc, char* argv[]) {
    enter_finch();
    Graph g = Graph{};
    N = make_weights_and_edges(argv[1], &g);
    edges = g.edges;
    damp = atof(argv[2]); //0.85
    beta_score = (1.0 - damp) / N;

    compile();
    auto ranks = std::get<3>(Main());
    jl_value_t *val = finch_exec("%s.lvl.lvl.val", ranks);
    double *data = (double*) jl_array_data(val);
    for(size_t i=0; i < N; i++) {
        std::cout << data[i] << "\n";
    }
    exit_finch();
}