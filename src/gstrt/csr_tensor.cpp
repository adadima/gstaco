//
// Created by Alexandra Dima on 04.03.2022.
//

#include "einsum_taco/gstrt/csr_tensor.h"

namespace einsum {
    int cmp(const void *a, const void *b) {
        return *((const int *) a) - *((const int *) b);
    }

    tensor_t *init_taco_tensor_t(int32_t order,
                                 int32_t *dimensions,
                                 format_t *mode_type) {
        tensor_t *t = (tensor_t *) malloc(sizeof(tensor_t));
        t->order = order;
        t->total_size = 1;
        for (int i = 0; i < order; i++) {
            t->total_size *= dimensions[i];
        }

        t->compressed_size = 1;
        if (order >= 2) {
            for (int i = 0; i < order - 2; i++) {
                t->compressed_size *= dimensions[i];
            }
        }

        t->dimensions = (int32_t *) malloc(order * sizeof(int32_t));
        for (int32_t i = 0; i < order; i++) {
            t->dimensions[i] = dimensions[i];
        }

        t->format = (format_t *) malloc(sizeof(format_t));
        return t;
    }

    void reduction(tensor_t *edges, tensor_t *contrib, tensor_t *rank) {
        // rank[i] = edges[i][j] * contrib[j] | j:(+, 0)
        compressed csr = edges->csr_data[0];
        int32_t rows = edges->dimensions[edges->order - 2];
        int32_t cols = edges->dimensions[edges->order - 1];
        for (int i = 0; i < rows; i++) {
            int init_j = 0;
            int start = csr.rows[i];
            int end = csr.rows[i + 1];
            for (int j = start; j < end; j++) {
                int val = csr.vals[j];
                init_j += val * contrib->normal_data[csr.cols[j]];
            }
            rank->normal_data[i] = init_j;
        }
    }

    void allocate(tensor_t *t) {
        for (int32_t i = 0; i < t->order; i++) {
            switch (*t->format) {
                case mode_dense:
                    t->normal_data = (uint8_t *) malloc(t->total_size * sizeof(uint8_t));
                    break;
                case mode_sparse:
                    einsum_iassert(t->order >= 2);
                    t->csr_data = (compressed *) malloc(t->compressed_size * sizeof(compressed));
                    int32_t rows = t->dimensions[t->order - 2];
                    for (int j = 0; j < t->compressed_size; j++) {
                        t->csr_data[j].rows = (int32_t *) malloc((rows + 1) * sizeof(int32_t));
                        t->csr_data[j].num_rows = rows;
                    }
                    break;
            }
        }
    }

    uint8_t get(tensor_t *tensor, int32_t *location) {
        uint8_t elt = 0;
        int index = location[0];

        switch (*tensor->format) {
            case mode_dense:
                for (int i = 1; i < tensor->order; i++) {
                    index = index * tensor->dimensions[i] + location[i];
                }
                elt = tensor->normal_data[index];
            case mode_sparse:
                for (int i = 1; i < tensor->order - 2; i++) {
                    index = index * tensor->dimensions[i] + location[i];
                }
                compressed csr = tensor->csr_data[index];
                int32_t row = location[tensor->order - 2];
                int32_t col = location[tensor->order - 1];
                int start = csr.rows[row];
                int end = csr.rows[row + 1];
                for (int j = start; j < end; j++) {
                    if (csr.cols[j] == col) {
                        elt = csr.vals[j];
                        break;
                    }
                }
        }
        return elt;
    }

    void set(tensor_t *tensor, int32_t *location, uint8_t val) {
        int index = location[0];

        switch (*tensor->format) {
            case mode_dense:
                for (int i = 1; i < tensor->order; i++) {
                    index = index * tensor->dimensions[i] + location[i];
                }
                tensor->normal_data[index] = val;
            case mode_sparse:
                if (tensor->order == 0) {
                    tensor->csr_data[0].cols[0] = 0;
                    tensor->csr_data[0].rows[0] = val;
                    return;
                }

                compressed csr;
                int32_t row;

                if (tensor->order == 1) {
                    csr = tensor->csr_data[0];
                    row = 0;
                } else {
                    for (int i = 1; i < tensor->order - 2; i++) {
                        index = index * tensor->dimensions[i] + location[i];
                    }
                    csr = tensor->csr_data[index];
                    row = location[tensor->order - 2];
                }

                int32_t col = location[tensor->order - 1];
                int start = csr.rows[row];
                int end = csr.rows[row + 1];
                int found = 0;
                for (int j = start; j < end; j++) {
                    if (j == col) {
                        csr.vals[j] = val;
                        found = 1;
                        break;
                    }
                }
                // insert new element in cols, and values => this will have shit performance
                if (!found) {
                    // modify rows
                    csr.rows[row + 1] += 1;

                    // modify cols
                    int8_t next = col;
                    for (int i = 0; i < csr.num_cols; i++) {
                        if (i > col) {
                            int8_t curr = csr.cols[i];
                            csr.cols[i] = next;
                            next = curr;
                        }
                    }
                    csr.num_cols += 1;
                    void *new_cols = realloc(csr.cols, csr.num_cols * sizeof(int32_t));
                    if (new_cols == NULL) {
                        // do something scary
                    } else {
                        csr.cols = (int32_t *) new_cols;
                        csr.cols[csr.num_cols - 1] = next;
                    }

                    //modify vals
                    int8_t next_v = val;
                    for (int i = 0; i < csr.num_cols; i++) {
                        if (i > col) {
                            int8_t curr = csr.vals[i];
                            csr.vals[i] = next;
                            next = curr;
                        }
                    }
                    csr.num_cols += 1;
                    void *new_vals = realloc(csr.cols, csr.num_cols * sizeof(uint8_t));
                    if (new_cols == NULL) {
                        // do something scary
                    } else {
                        csr.vals = (uint8_t *) new_vals;
                        csr.cols[csr.num_cols - 1] = next_v;
                    }
                }
        }
    }

    void deallocate(tensor_t *t) {
        switch (*t->format) {
            case mode_dense:
                free(t->normal_data);
            case mode_sparse:
                for (int i = 0; i < t->compressed_size; i++) {
                    compressed csr = t->csr_data[i];
                    if (csr.num_rows > 0) {
                        free(t->csr_data[i].rows);
                    }
                    if (csr.num_cols > 0) {
                        free(t->csr_data[i].cols);
                        free(t->csr_data[i].vals);
                    }
                }
                free(t->csr_data);
        }
        free(t->dimensions);
        free(t->format);
        free(t);
    }
}