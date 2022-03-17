//
// Created by Alexandra Dima on 21.02.2022.
//

#include "../../../tmp/codegen/bc.h"
#include <string>
#include <fstream>

int N;
int P;
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
    P = 1000000000;
    N = std::get<0>(tensors);
    edges = std::get<1>(tensors);

    weights = std::get<2>(tensors);

    source = atoi(argv[2]);

    auto result = BC();
    auto deps = std::get<0>(result);
    std::string output;
    for(int i=0; i < deps.total_size; i++) {
        std::cout << deps.data[i] << std::endl;
        output += std::to_string(deps.data[i]) + "\n";
    }
    writeStringToFile(argv[3], output);
}