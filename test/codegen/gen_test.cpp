//
// Created by Alexandra Dima on 11/15/21.
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir.h"
#include "einsum_taco/ir/cleanup.h"
#include "einsum_taco/codegen/codegen_visitor.h"
#include "einsum_taco/ir/dump_ast.h"
#include "einsum_taco/gstrt/tensor.h"
#include <einsum_taco/parser/heading.h>
#include "../utils.h"
#include <filesystem>

using namespace std;

namespace fs = std::filesystem;

class GenTest : public testing::TestWithParam<std::tuple<std::string, std::string, bool>> {
public:
    std::stringstream oss_cpp;
    std::stringstream oss_h;
    CodeGenVisitor generator;
    DumpAstVisitor printer;


    GenTest() : generator(&oss_cpp, &oss_h, std::get<0>(GetParam())) {}

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

    static void writeStringToFile(const std::string& filename, const std::string& generated_code) {
        std::ofstream out(filename);
        out << generated_code;
        out.close();
    }

    static std::string get_runtime_dir() {
        return {GSTACO_RUNTIME};
    }

    static std::string readDataIntoString(const std::string& path) {
        return readFileIntoString(path);
    }

    static std::string get_tensor_template() {
        return readFileIntoString(get_runtime_dir() + "tensor.h");
    }

    std::string tensor_template = get_tensor_template();

    static std::string test_name_to_input_file(const std::string& test_name) {
        return get_test_data_dir() + "codegen/inputs/" + test_name + ".txt";
    }

    static std::string test_name_to_expected_file(const std::string& test_name) {
        return get_test_data_dir() + "codegen/outputs/" + test_name + ".cpp";
    }

    static std::string test_name_to_header(const std::string& test_name) {
        return get_tmp_dir_name() + test_name + ".h";
    }

    static std::string get_tmp_dir_name() {
        return get_test_dir() + "tmp/codegen/";
    }

    static std::string test_name_to_tmp_input(const std::string& test_name) {
        return get_tmp_dir_name() + test_name + ".cpp";
    }

    static std::string test_name_to_tmp_output(const std::string& test_name) {
        return get_tmp_dir_name() + test_name + ".out";
    }

    static std::string test_name_to_driver(const std::string& test_name) {
        return get_test_data_dir() + "codegen/drivers/driver_" + test_name + ".cpp";
    }

    std::tuple<std::string, std::string> get_generated_code(const std::string& test_name) {
        // read input
        auto input = readDataIntoString(test_name_to_input_file(test_name));

        // parse
        auto mod = std::make_shared<Module>(parse(input));

        // cleanup
        auto new_module = apply_default_rewriters(mod);

        // print ast for debug
//        new_module->accept(&printer);
//        std::cout << printer.ast;

        // code generation
        new_module->accept(&generator);
        return std::tuple{oss_cpp.str(), oss_h.str()};
    }

    // borrowed from https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
    static std::tuple<int, std::string> exec(const char* cmd) {
        std::array<char, 128> buffer{};
        std::string result;

        // start process to run cmd
        FILE* pipe = popen(cmd, "r");
        if (!pipe) {
            std::cerr << "Couldn't open process." << std::endl;
            throw std::runtime_error("popen() failed!");
        }

        // read all stdout to buffer
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }

        // get return code
        auto returnCode = pclose(pipe);

        return {returnCode, result};
    }

    static void create_dir(const std::string& dir) {
        // if directory does not exist, create it
        std::error_code ec;
        bool success = fs::create_directories(dir, ec);
    }

    void cleanup(const std::string& in, const std::string& out, const std::string& header) {
        fs::remove(in);
        fs::remove(out);
        fs::remove(header);
    }

    void assert_compiles(const std::string& test_name, const std::string& compiler_path, bool add_main = true) {
        create_dir(get_tmp_dir_name());

        std::string tmp_header = test_name_to_header(test_name);
        std::string tmp_in = test_name_to_tmp_input(test_name);
        std::string tmp_out = test_name_to_tmp_output(test_name);
        std::string driver = test_name_to_driver(test_name);

        // generate code from input file and add dummy main function
        std::string code;
        std::string header;
        std::tie(code, header) = get_generated_code(test_name);
        if (add_main) {
            code += R"(
int main() {}
)";
        }
        // write generated code to temporary cpp file
        writeStringToFile(tmp_header, header);

        // write generated code to temporary cpp file
        writeStringToFile(tmp_in, code);

        // compile the temporary cpp file with the given compiler
        std::string cmd = compiler_path + " -o " + tmp_out + " -std=c++17 " + tmp_in;
        if (!add_main) {
            cmd += " " + driver;
        }
        int status_code;
        std::string output;
        std::tie(status_code, output) = exec(cmd.c_str());

        // remove generated files
        cleanup(tmp_in, tmp_out, tmp_header);

        // check compilation process finished successfully
        EXPECT_EQ(status_code, 0);
    }

    void assert_generated(const std::string& test_name) {
        auto expected_filename = test_name_to_expected_file(test_name);

        auto expected = tensor_template + readDataIntoString(expected_filename);
        std::string code;
        std::string header;
        std::tie(code, header) = get_generated_code(test_name);
        EXPECT_EQ(code, expected);
    }
};

TEST_P(GenTest, Compilation) {
    auto params = GetParam();
    auto test_name = std::get<0>(params);
    auto compiler = std::get<1>(params);
    auto add_main = std::get<2>(params);
    assert_compiles(test_name, compiler, add_main);
}

INSTANTIATE_TEST_CASE_P(
        GenTestSuite,
        GenTest,
        ::testing::Values(
                make_tuple("definition1", get_compiler_path(), true),
                make_tuple("definition2", get_compiler_path(), true),
                make_tuple("definition3", get_compiler_path(), true),
                make_tuple("definition4", get_compiler_path(), true),
                make_tuple("call", get_compiler_path(), true),
                make_tuple("call_condition1", get_compiler_path(), true),
                make_tuple("call_condition2", get_compiler_path(), true),
                make_tuple("call_repeat1", get_compiler_path(), true),
                make_tuple("call_repeat2", get_compiler_path(), true),
                make_tuple("call_repeat3", get_compiler_path(), true),
                make_tuple("call_repeat4", get_compiler_path(), true),
                make_tuple("call_repeat5", get_compiler_path(), true),
                make_tuple("outer_loop_var1", get_compiler_path(), true),
                make_tuple("outer_loop_var2", get_compiler_path(), true),
                make_tuple("bfs_step", get_compiler_path(), true),
                make_tuple("pagerank", get_compiler_path(), false)
        ));