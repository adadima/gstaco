//
// Created by Alexandra Dima on 17.02.2022.
//

#include "einsum_taco/gstrt/tensor.h"

#include <vector>
#include <string>
#include <sstream>

static std::string readFileIntoString(const std::string& path) {
    FILE *fp = fopen(path.c_str(), "r");
    if (fp == nullptr) {
        std::cout << "Failed to open file for reading " << path << std::endl;
        std::abort();
    }
    auto size = fs::file_size(path);
    std::string contents = std::string(size, 0);
    fread(contents.data(), 1, size, fp);
    fclose(fp);
    return contents;
}

std::vector<std::string> getLines(const std::string& content) {
    std::stringstream ss_file(content);
    std::string line;

    auto lines = std::vector<std::string>();

    while(std::getline(ss_file, line,'\n')){
        lines.push_back(line);
    }
    return lines;
}

std::tuple<int, Tensor<int, 2>, Tensor<float, 2>> loadEdgesFromFile(const std::string& filename) {
    auto content = readFileIntoString(filename);

    auto lines = getLines(content);
    int size = lines.size();

    Tensor<int, 2> edges({size, size});
    edges.allocate();

    Tensor<float, 2> weights({size, size});
    weights.allocate();

    for (int i=0; i < size; i++) {
        auto line = lines[i];
        std::stringstream ss(line);
        std::string num;

        std::getline(ss, num, ' ');
        auto dst = std::atoi(num.c_str()) - 1;

        std::getline(ss, num, ' ');
        auto src = std::atoi(num.c_str()) - 1;

        edges.at({dst, src}) = 1;

        std::getline(ss, num, ' ');
        auto w= std::stof(num.c_str());

        weights.at({dst, src}) = w;
    }

    return std::tuple{size, edges, weights};

}