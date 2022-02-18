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
    std::cout << "N: " << N << std::endl;
    edges = std::get<1>(tensors);
    std::cout << "Edges: " << std::endl;
    for (int i=0; i < edges.total_size; i++) {
        std::cout << edges.data[i] << std::endl;
    }
    for(int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            std::cout << "Edge from " << j << " to " << i << " with weight " << edges.at({i, j}) << std::endl;
            }
    }

    weights = std::get<2>(tensors);

    damp = atof(argv[2]); //0.85

    beta_score = (1.0 - damp) / N;

    auto result = PageRank();
    auto ranks = std::get<3>(result);

    for(int i=0; i < ranks.total_size; i++) {
        std::cout << ranks.data[i] << std::endl;
    }
}