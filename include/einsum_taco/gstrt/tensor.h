#include <array>
#include <iostream>

template<class T, int num_dims>
struct Tensor {
    T* data;
    int total_size;
    std::array<int, num_dims> dims;

    explicit Tensor(std::array<int, num_dims> dims): dims(dims) {
        total_size = 1;
        for (auto dim: dims) {
            total_size *= dim;
        }
    }

    void allocate() {
        data = new T[total_size];
    }

    void deallocate() {
        delete[] data;
    }

    //TODO:: optimize this to be placed inline
    T& at(std::array<int, num_dims> idx) {
        int index = idx[0];
        for (int i = 1; i < num_dims; i++) {
            index = index * dims[i - 1] + idx[i];
        }
        return data[index];
    }

    T& at() {
        // assert(total_size == 1);   need to enforce this somehow
        return data[0];
    }
};

// usage:
//Tensor<int, 3> t({2, 2, 2});
// t.allocate();
//t.at({1, 0, 1}) = 5;
// t.deallocate();
