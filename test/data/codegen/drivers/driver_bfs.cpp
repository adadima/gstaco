//
// Created by Alexandra Dima on 21.02.2022.
//

#include "../../../tmp/codegen/bfs.h"
#include <string>
#include <fstream>

int N;
int source;
Tensor<int, 2> edges({}, mode_sparse);
Tensor<float, 2> weights({}, mode_sparse);

static void writeStringToFile(const std::string& filename, const std::string& generated_code) {
    std::ofstream out(filename);
    out << generated_code;
    out.close();
}

int main(int argc, char* argv[]) {
    std::tuple<int, Tensor<int, 2>, Tensor<float, 2>> tensors = loadEdgesFromFile(argv[1]);

    N = std::get<0>(tensors);
    edges = std::get<1>(tensors);

    weights = std::get<2>(tensors);

    source = atoi(argv[2]);

    auto result = BFS();
    auto parents = std::get<0>(result);
    std::string output;
    for(int i=0; i < parents.total_size; i++) {
        std::cout << parents.data[i] << std::endl;
        output += std::to_string(parents.data[i]) + "\n";
    }
    writeStringToFile(argv[3], output);
}