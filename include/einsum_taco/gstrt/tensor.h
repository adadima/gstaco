#include <array>
#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

template<class T, int num_dims>
struct Tensor {
    static_assert(std::is_trivial_v<T>);
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
        std::fill(data, data + total_size, T{});
    }

    void deallocate() {
        delete[] data;
    }

    //TODO:: optimize this to be placed inline
    T& at(std::array<int, num_dims> idx) {
        int index = idx[0];
        for (int i = 1; i < num_dims; i++) {
            index = index * dims[i] + idx[i];
        }
        return data[index];
    }

    T& at() {
        // assert(total_size == 1);   need to enforce this somehow
        return data[0];
    }

    friend bool operator==(const Tensor<T, num_dims>& lhs, const T& rhs) {
        for (int i=0; i < lhs.total_size; i++) {
            if (lhs.data[i] != rhs) {
                return false;
            }
        }
        return true;
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
        for (int i=0; i < tensor.total_size; i++) {
            oss << tensor.data[i] << " ";
        }
        oss << std::endl;
        return oss;
    }
};

// usage:
//Tensor<int, 3> t({2, 2, 2});
// t.allocate();
//t.at({1, 0, 1}) = 5;
// t.deallocate();

std::tuple<int, Tensor<int, 2>, Tensor<float, 2>> loadEdgesFromFile(const std::string& filename);
