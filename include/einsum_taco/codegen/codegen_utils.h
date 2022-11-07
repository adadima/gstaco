//
// Created by Alexandra Dima on 02.11.2022.
//

#ifndef EINSUM_TACO_CODEGEN_UTILS_H
#define EINSUM_TACO_CODEGEN_UTILS_H
//
// Created by Alexandra Dima on 02.11.2022.
//
#include<iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <streambuf>

std::string readFileIntoString(const std::string& path);

std::string get_runtime_include_dir();

std::string get_runtime_src_dir();

std::string parse_variable_name(const std::string& var);
#endif //EINSUM_TACO_CODEGEN_UTILS_H
