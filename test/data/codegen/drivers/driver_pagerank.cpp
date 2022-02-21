//
// Created by Alexandra Dima on 16.02.2022.
//

#include <string>
#include "../../../tmp/codegen/pagerank.h"

int N;
Tensor<int, 2> edges({});
Tensor<float, 2> weights({});
float damp;
float beta_score;

int main(int argc, char* argv[]) {
    std::tuple<int, Tensor<int, 2>, Tensor<float, 2>> tensors = loadEdgesFromFile(argv[1]);

    N = std::get<0>(tensors);
    edges = std::get<1>(tensors);

    weights = std::get<2>(tensors);

    damp = atof(argv[2]); //0.85

    beta_score = (1.0 - damp) / N;

    auto result = PageRank();
    auto ranks = std::get<3>(result);

    for(int i=0; i < ranks.total_size; i++) {
        std::cout << ranks.data[i] << std::endl;
    }
}