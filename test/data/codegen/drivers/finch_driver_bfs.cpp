//
// Created by Alexandra Dima on 30.10.2022.
//

#include "../../../tmp/codegen/bfs.h"
#include <string>
#include <fstream>
#include "julia.h"
#include <iostream>

int N;
int source;
jl_value_t* edges;

int main(int argc, char* argv[]) {
    enter_finch();
    Graph g = Graph{};
    N = make_weights_and_edges(argv[1], &g);
    edges = g.edges;
    source = std::atoi(argv[2]);

    compile();
    auto res = Main();
    auto parents = std::get<0>(res);
    jl_value_t *val = finch_exec("%s.lvl.lvl.val", parents);
    int *data = (int*) jl_array_data(val);
    for(int i=1; i < N+1; i++) {
        finch_exec("println((%s.lvl.lvl.val)[%s])", parents, finch_Int64(i));
    }
    exit_finch();
}