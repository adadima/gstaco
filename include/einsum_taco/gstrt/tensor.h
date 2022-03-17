#include <array>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cassert>

namespace fs = std::filesystem;

typedef enum {
    mode_dense, mode_sparse
} format_t;

using namespace std;

template<class T>
struct CSR {
    vector<int> rows;

    vector<int> cols;

    vector<T>   vals;

    CSR() {}

    void allocate(int num_rows) {
        rows.assign(num_rows, 0);
    }

    CSR(int num_rows): rows(num_rows, 0) {}

    friend bool operator==(const CSR<T>& lhs, const T& rhs) {
        int num_rows = lhs.rows.size();
        for (int r=0; r < num_rows - 1; r++) {
            int start = lhs.rows[r];
            int end = lhs.rows[r + 1];
            for (int j = start; j < end; j++) {
                if (lhs.cols[j] != rhs) {
                    return false;
                }
            }
        }
        return true;
    }

    friend std::ostream& operator<<(std::ostream &oss, const CSR<T>& csr) {
        int num_rows = csr.rows.size();
        for (int r=0; r < num_rows - 1; r++) {
            int start = csr.rows[r];
            int end = csr.rows[r + 1];
            for (int j = start; j < end; j++) {
                oss << "(" << r << ", " << csr.cols[j] << "):" << csr.vals[j] << std::endl;
            }
        }
        return oss;
    }
};

template<class T, int num_dims>
struct Tensor {
    static_assert(std::is_trivial_v<T>);
    T* data;
    int32_t total_size;
    std::array<int, num_dims> dims;
    format_t format;
    int compressed_size;        // number of compressed structs to allocate (csr arrays)
    CSR<T> *csr_data;               // array of compressed csr arrays for the last two dimensions, only populated in compressed format

    explicit Tensor(std::array<int, num_dims> dims, format_t format): dims(dims), format(format) {
        switch (format) {
            case mode_dense:
                total_size = 1;
                for (auto dim: dims) {
                    total_size *= dim;
                }
            case mode_sparse:
                compressed_size = 1;
                if (num_dims >= 2) {
                    for (int i = 0; i < num_dims - 2; i++) {
                        compressed_size *= dims[i];
                    }
                }
        }
    }

    void allocate() {

        switch (format) {
                case mode_dense:
                    data = new T[total_size];
                    std::fill(data, data + total_size, T{});
                    break;
                case mode_sparse:
                    assert(num_dims >= 2);
                    csr_data = new CSR<T>[compressed_size];
                    int32_t rows = dims[num_dims - 2];
                    for (int i=0; i < compressed_size; i++) {
                        csr_data[i].allocate(rows + 1);
                    }
                    break;
        }
    }

    void deallocate() {
        switch (format) {
            case mode_dense:
                delete []data;
            case mode_sparse:
                delete []csr_data;
        }
    }

    //TODO:: optimize this to be placed inline
    T get(std::array<int, num_dims> idx) {
        int index = 0;
        switch(format) {
            case mode_dense:
                for (int i = 0; i < num_dims; i++) {
                    index = index * dims[i] + idx[i];
                }
                return data[index];
            case mode_sparse:
                for (int i = 0; i < num_dims - 2; i++) {
                    index = index * dims[i] + idx[i];
                }
                CSR<T> csr = csr_data[index];
                int32_t row = idx[num_dims - 2];
                int32_t col = idx[num_dims - 1];
                int start = csr.rows[row];
                int end = csr.rows[row + 1];
                for (int j = start; j < end; j++) {
                    if (csr.cols[j] == col) {
                        return csr.vals[j];
                    }
                }
                return T{};
        }
    }

    void set(std::array<int, num_dims> idx, T val) {
        int index = 0;

        switch (format) {
            case mode_dense:
                for (int i = 0; i < num_dims; i++) {
                    index = index * dims[i] + idx[i];
                }
                data[index] = val;
                return;
            case mode_sparse:

                for (int i = 0; i < num_dims - 2; i++) {
                    index = index * dims[i] + idx[i];
                }
//                std::cout << "CSR index: " << index << std::endl;
                CSR<T>& csr = csr_data[index];
                int row = idx[num_dims - 2];
                int col = idx[num_dims - 1];
//                std::cout << "Inserting value " << val << " at (" << row << ", " << col << ")" << std::endl;
                int start = csr.rows[row];
                int end = csr.rows[row + 1];

                for (int j = start; j < end; j++) {
                    if (csr.cols[j] == col) {
                        csr.vals[j] = val;
                        return;
                    }
                }
//                std::cout << "Did not find coords in CSR, adding new column" << std::endl;
                int col_idx = start;
                for (int j = start; j < end; j++) {
                    if (csr.cols[j] < col) {
                        col_idx += 1;
                    }
                    if (csr.cols[j] > col) {
                        break;
                    }
                }
//                std::cout << "insert col " << col << " at position " << col_idx << std::endl;
                // modify cols
                csr.cols.insert(csr.cols.begin() + col_idx, col);
                // modify vals
                csr.vals.insert(csr.vals.begin() + col_idx, val);
                // modify rows
                for (int r = row+1; r < csr.rows.size(); r++) {
                    csr.rows[r] += 1;
                }

//                std::cout << "New CSR: " << csr;
            }
    }

    T get() {
        assert(format == mode_dense);
        assert(total_size == 1);
        return data[0];
    }

    void set(T val) {
        assert(format == mode_dense);
        assert(total_size == 1);
        data[0] = val;
    }

    friend bool operator==(const Tensor<T, num_dims>& lhs, const T& rhs) {
        switch(lhs.format) {
            case mode_dense:
                for (int i=0; i < lhs.total_size; i++) {
                    if (lhs.data[i] != rhs) {
                        return false;
                    }
                }
                return true;
            case mode_sparse:
                if (rhs == T{}) {
                    for(int i=0; i < lhs.compressed_size; i++) {
                        auto csr = lhs.csr_data[i];
                        if (csr.cols.size() > 0) {
                            return false;
                        }
                    }
                    return true;
                } else {
                    int32_t cols = lhs.dims[num_dims - 1];
                    for(int i=0; i < lhs.compressed_size; i++) {
                        auto csr = lhs.csr_data[i];
                        if (csr.cols.size() == cols) {
                            return csr == rhs;
                        }
                    }
                    return true;
                }
        }
    }

    friend bool operator!=(const Tensor<T, num_dims>& lhs, const T& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator==(const T& lhs, const Tensor<T, num_dims>& rhs) {
        return rhs == lhs;
    }

    friend bool operator!=(const T& lhs, const Tensor<T, num_dims>& rhs) {
        return !(rhs == lhs);
    }

    friend std::ostream& operator<<(std::ostream &oss, const Tensor<T, num_dims> tensor) {
        switch(tensor.format) {
            case mode_dense:
                for (int i=0; i < tensor.total_size; i++) {
                    oss << tensor.data[i] << " ";
                }
            case mode_sparse:
                for (int i=0; i < tensor.compressed_size; i++) {
                    oss << tensor.csr_data[i];
                }
        }
        oss << std::endl;
        return oss;

    }
};

std::tuple<int, Tensor<int, 2>, Tensor<float, 2>> loadEdgesFromFile(const std::string& filename);
