//
// Created by Alexandra Dima on 30.10.2022.
//

#ifndef EINSUM_TACO_RUNTIME_H
#define EINSUM_TACO_RUNTIME_H

#include <julia.h>
#include "finch.h"
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Graph {
    jl_value_t* weights;
    jl_value_t* edges;
};

int make_weights_and_edges(const char *graph_name, struct Graph* graph);

void enter_finch();

void exit_finch();
#ifdef __cplusplus
}
#endif

#endif //EINSUM_TACO_RUNTIME_H
