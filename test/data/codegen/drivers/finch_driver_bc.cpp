//
// Created by Alexandra Dima on 30.10.2022.
//


#include "../../../tmp/codegen/bc.h"
#include <string>
#include <fstream>
#include "julia.h"
#include "einsum_taco/gstrt/runtime.h"

int N;
int P;
int source;
jl_value_t* edges;

int main(int argc, char* argv[]) {
    enter_finch();

    Graph g = Graph{};
    N = make_weights_and_edges(argv[1], &g);
    source = std::atoi(argv[2]);

    P = 1000000000;
    edges = g.edges;
    source = atoi(argv[2]);

    compile();
    auto res = Main();
    auto deps = std::get<0>(res);
    for(int i=1; i < N+1; i++) {
        finch_exec("println((%s.lvl.lvl.val)[%s])", deps, finch_Int64(i));
    }

    exit_finch();
}