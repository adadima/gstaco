//
// Created by Alexandra Dima on 28.02.2022.
//

#ifndef EINSUM_TACO_CSR_TENSOR_H
#define EINSUM_TACO_CSR_TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include "einsum_taco/base/assert.h"

namespace einsum {

    typedef enum {
        mode_dense, mode_sparse
    } format_t;

    typedef struct {
        int32_t *rows;
        int32_t num_rows;

        int32_t *cols;
        int32_t num_cols;

        uint8_t *vals;
    } compressed;

    typedef struct {
        int32_t order;                  // tensor order ( num dimensions )
        int32_t total_size;             // tensor total size
        int32_t compressed_size;        // number of compressed sturcts to allocate (csr arrays)
        int32_t *dimensions;             // tensor dimensions
        compressed *csr_data;               // array of compressed csr arrays for the last two dimensions, only populated in compressed format
        uint8_t *normal_data;            // array of all the entries, concatenated in row-major order, only populated in uncompressed format
        format_t *format;
    } tensor_t;

    int cmp(const void *a, const void *b);

    tensor_t *init_taco_tensor_t(int32_t order,
                                 int32_t *dimensions,
                                 format_t *mode_type);

    void reduction(tensor_t *edges, tensor_t *contrib, tensor_t *rank);

    void allocate(tensor_t *t);

    uint8_t get(tensor_t *tensor, int32_t *location);

    void set(tensor_t *tensor, int32_t *location, uint8_t val);

    void deallocate(tensor_t *t);
}

#endif //EINSUM_TACO_CSR_TENSOR_H
