#include <array>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cassert>
#include "taco/tensor.h"

namespace fs = std::__fs::filesystem;

static std::string readFileIntoString(const std::string& path) {
    FILE *fp = fopen(path.c_str(), "r");
    if (fp == nullptr) {
        std::cout << "Failed to open file for reading " << path << std::endl;
        std::abort();
    }
    auto size = fs::file_size(path);
    std::string contents = std::string(size, 0);
    fread((void *) contents.data(), 1, size, fp);
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

inline std::string get_test_dir() {
    return {EINSUM_TACO_TEST_DATADIR};
}

std::tuple<int, Tensor<int>, Tensor<int>> loadEdgesFromFile(const std::string& filename) {
    auto content = readFileIntoString(filename);

    auto lines = getLines(content);
    int size = std::atoi(lines[0].c_str());

    Format csr({Dense,Sparse});
    Tensor<int> edges("edges", {size, size}, csr);
    Tensor<int> weights("weights", {size, size}, csr);


    for (int i=1; i < lines.size(); i++) {
        auto line = lines[i];
        std::stringstream ss(line);
        std::string num;

        std::getline(ss, num, ' ');
        auto dst = std::atoi(num.c_str()) - 1;

        std::getline(ss, num, ' ');
        auto src = std::atoi(num.c_str()) - 1;
        edges.insert({dst, src}, 1);

        std::getline(ss, num, ' ');
        auto w= std::atoi(num.c_str());
        weights.insert({dst, src}, w);
    }

    return {size, edges, weights};
}