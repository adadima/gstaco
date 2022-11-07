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

    res = finch_eval("or(x,y) = x == 1|| y == 1\n\
function choose(x, y)\n\
    if x != 0\n\
        return x\n\
    else\n\
        return y\n\
    end\n\
end");

    res = finch_eval("@slots a b c d e i j Finch.add_rules!([\n\
    (@rule @f(@chunk $i a (b[j...] <<min>>= $d)) => if Finch.isliteral(d) && i ∉ j\n\
        @f (b[j...] <<min>>= $d)\n\
    end),\n\
    (@rule @f(@chunk $i a @multi b... (c[j...] <<min>>= $d) e...) => begin\n\
        if Finch.isliteral(d) && i ∉ j\n\
            @f @multi (c[j...] <<min>>= $d) @chunk $i a @f(@multi b... e...)\n\
        end\n\
    end),\n\
    \n\
    (@rule @f($or(false, $a)) => a),\n\
    (@rule @f($or($a, false)) => a),\n\
    (@rule @f($or($a, true)) => true),\n\
    (@rule @f($or(true, $a)) => true),\n\
    \n\
    (@rule @f(@chunk $i a (b[j...] <<$choose>>= $d)) => if Finch.isliteral(d) && i ∉ j\n\
        @f (b[j...] <<$choose>>= $d)\n\
    end),\n\
    (@rule @f(@chunk $i a @multi b... (c[j...] <<choose>>= $d) e...) => begin\n\
        if Finch.isliteral(d) && i ∉ j\n\
            @f @multi (c[j...] <<choose>>= $d) @chunk $i a @f(@multi b... e...)\n\
        end\n\
    end),\n\
    (@rule @f($choose(0, $a)) => a),\n\
    (@rule @f($choose($a, 0)) => a),\n\
    (@rule @f(@chunk $i a (b[j...] <<$or>>= $d)) => if Finch.isliteral(d) && i ∉ j\n\
        @f (b[j...] <<$or>>= $d)\n\
    end),\n\
    (@rule @f(@chunk $i a @multi b... (c[j...] <<$or>>= $d) e...) => begin\n\
        if Finch.isliteral(d) && i ∉ j\n\
            @f @multi (c[j...] <<$or>>= $d) @chunk $i a @f(@multi b... e...)\n\
        end\n\
    end),\n\
])\n\
\n\
Finch.register()");
}


void exit_finch() {
    finch_finalize();
}



