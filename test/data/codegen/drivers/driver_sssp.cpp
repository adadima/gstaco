//
// Created by Alexandra Dima on 21.02.2022.
//

#include "../../../tmp/codegen/sssp.h"
#include <string>

int N;
int P;
int source;
Tensor<int, 2> edges({});
Tensor<float, 2> weights({});

int main(int argc, char* argv[]) {
    std::tuple<int, Tensor<int, 2>, Tensor<float, 2>> tensors = loadEdgesFromFile(argv[1]);

    N = std::get<0>(tensors);
    edges = std::get<1>(tensors);

    weights = std::get<2>(tensors);

    source = atoi(argv[2]);
    P = atoi(argv[3]);

    auto result = SSSP();
    auto dists = std::get<0>(result);
    for(int i=0; i < dists.total_size; i++) {
        std::cout << dists.data[i] << std::endl;
    }
}