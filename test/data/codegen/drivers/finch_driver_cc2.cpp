//
// Created by Alexandra Dima on 30.10.2022.
//
#include "../../../tmp/codegen/cc2.h"
#include <string>
#include <fstream>
#include "julia.h"

int N;
jl_value_t* edges;

int main(int argc, char* argv[]) {
    enter_finch();

    Graph g = Graph{};
    N = make_weights_and_edges(argv[1], &g);
    edges = g.edges;

    compile();
    auto res = Main();
    auto ids = std::get<0>(res);
    for(int i=1; i < N+1; i++) {
        finch_exec("println((%s.lvl.lvl.val)[%s])", ids, finch_Int64(i));
    }

    exit_finch();
}