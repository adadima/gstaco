//
// Created by Alexandra Dima on 30.10.2022.
//
#include "einsum_taco/gstrt/runtime.h"
#include <stdio.h>


int make_weights_and_edges(const char* graph_name, struct Graph* graph) {
    char code1[1000];
    sprintf(code1, "matrix = copy(transpose(MatrixMarket.mmread(\"./graphs/%s\")))\n\
        (n, m) = size(matrix)\n\
        @assert n == m\n\
        nzval = ones(size(matrix.nzval, 1))\n\
        Finch.Fiber(\n\
                 Dense(n,\n\
                 SparseList(n, matrix.colptr, matrix.rowval,\n\
                 Element{0}(nzval))))", graph_name);
    graph->edges = finch_eval(code1);
    printf("Loaded edges\n");

    char code2[1000];
    sprintf(code2, "matrix = copy(transpose(MatrixMarket.mmread(\"./graphs/%s\")))\n\
        (n, m) = size(matrix)\n\
        @assert n == m\n\
        Finch.Fiber(\n\
                 Dense(n,\n\
                 SparseList(n, matrix.colptr, matrix.rowval,\n\
                 Element{0}(matrix.nzval))))", graph_name);
    graph->weights = finch_eval(code2);
    printf("Loaded weights\n");

    int* n = (int*) finch_exec("%s.lvl.I", graph->edges);
    return *n;
}


void enter_finch() {
    finch_initialize();

    jl_value_t* res = finch_eval("using RewriteTools\n\
    using Finch.IndexNotation\n\
    using SparseArrays\n\
     using MatrixMarket\n\
    ");
}


void exit_finch() {
    finch_finalize();
}



