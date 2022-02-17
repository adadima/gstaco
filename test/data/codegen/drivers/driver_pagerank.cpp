//
// Created by Alexandra Dima on 16.02.2022.
//

#include <string>
#include "../../../tmp/codegen/pagerank.h"

int main(int argc, char* argv[]) {
    int N = atoi(argv[1]);

    float damp = atoi(argv[2]); //0.85

    float beta_score = (1.0 - damp) / N;

    Tensor<int, 2> edges({N, N});
    edges.allocate();

    auto result = PageRank();
    auto ranks = std::get<3>(result);

    for(int i=0; i < edges.total_size; i++) {
        std::cout << ranks.data[i] << std::endl;
    }
}