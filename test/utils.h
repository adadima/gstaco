//
// Created by Alexandra Dima on 19.12.2021.
//

#ifndef EINSUM_TACO_UTILS_H
#define EINSUM_TACO_UTILS_H

#include <string_view>
#include <fstream>
#include <streambuf>

inline einsum::Module parse(std::string_view code) {
    char temp_name[17];
    strcpy(temp_name, "dump_test_XXXXXX");
    int f = mkstemp(temp_name);

    if (f == -1) {
        throw std::system_error(errno, std::system_category());
    }
    auto temp = fdopen(f, "r+");
    auto num_written = fwrite(code.data(), 1, code.size(), temp);

    if (num_written != code.size()) {
        throw std::runtime_error("Did not write correct bytes");
    }
    auto status = fseek(temp, 0, SEEK_SET);
    auto module = parse_module(temp);

    fclose(temp);
    close(f);
    return module;
}

#ifndef EINSUM_TACO_TEST_DATADIR
#error "Did not receive test data dir."
#endif

#ifndef TEST_CXX_COMPILER
#error "Did not set c++ compiler variable."
#endif

inline std::string get_test_data_dir() {
    return {EINSUM_TACO_TEST_DATADIR};
}

inline std::string get_test_dir() {
    return {EINSUM_TACO_TEST_DIR};
}

inline std::string get_compiler_path() {
    return {TEST_CXX_COMPILER};
}

inline std::string get_julia_include_dir() {
    return {JULIA_INCLUDE_DIR};
}

inline std::string get_julia_lib_dir() {
    return {JULIA_LIB_DIR};
}

inline std::string get_finch_embed_dir() {
    return {FINCH_DIR};
}

static std::string readFileIntoString(const std::string& path) {
    std::ifstream istrm(path);

    if (!istrm.is_open()) {
        std::cout << "Failed to open file for reading " << path << std::endl;
        std::abort();
    }

    std::stringstream buffer;
    buffer << istrm.rdbuf();

    return buffer.str();
}
#endif //EINSUM_TACO_UTILS_H