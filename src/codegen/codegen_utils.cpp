//
// Created by Alexandra Dima on 02.11.2022.
//
#include<iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <streambuf>

std::string readFileIntoString(const std::string& path) {
    std::ifstream istrm(path);

    if (!istrm.is_open()) {
        std::cout << "Failed to open file for reading " << path << std::endl;
        std::abort();
    }

    std::stringstream buffer;
    buffer << istrm.rdbuf();

    return buffer.str();
}

std::string get_runtime_include_dir() {
    return {INCLUDE_GSTACO_RUNTIME};
}

std::string get_runtime_src_dir() {
    return {SRC_GSTACO_RUNTIME};
}