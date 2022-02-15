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

class GenTest : public testing::Test {
public:
    std::stringstream oss;
    CodeGenVisitor generator;
    DumpAstVisitor printer;


    GenTest() : generator(oss, "test") {}

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

    static std::string get_tmp_dir_name() {
        return get_test_dir() + "tmp/codegen/";
    }

    static std::string test_name_to_tmp_input(const std::string& test_name) {
        return get_tmp_dir_name() + test_name + ".cpp";
    }

    static std::string test_name_to_tmp_output(const std::string& test_name) {
        return get_tmp_dir_name() + test_name + ".out";
    }

    std::string get_generated_code(const std::string& test_name) {
        // read input
        auto input = readDataIntoString(test_name_to_input_file(test_name));

        // parse
        auto mod = std::make_shared<Module>(parse(input));

        // cleanup
        auto new_module = apply_default_rewriters(mod);

        // print ast for debug
        new_module->accept(&printer);
        std::cout << printer.ast;

        // code generation
        new_module->accept(&generator);
        return oss.str();
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

    void cleanup(const std::string& tmp_in, const std::string& tmp_out) {
        fs::remove(tmp_in);
        fs::remove(tmp_out);
    }

    void assert_compiles(const std::string& test_name, const std::string& compiler_path) {
        create_dir(get_tmp_dir_name());

        std::string tmp_in = test_name_to_tmp_input(test_name);
        std::string tmp_out = test_name_to_tmp_output(test_name);

        // generate code from input file and add dummy main function
        auto code = get_generated_code(test_name) +
                R"(
int main() {}
)";
        // write generated code to temporary cpp file
        writeStringToFile(tmp_in, code);

        // compile the temporary cpp file with the given compiler
        std::string cmd = compiler_path + " -o " + tmp_out + " -std=c++17 " + tmp_in;
        int status_code;
        std::string output;
        std::tie(status_code, output) = exec(cmd.c_str());

        // remove generated files
        // cleanup(tmp_in, tmp_out);

        // check compilation process finished successfully
        EXPECT_EQ(status_code, 0);
    }

    void assert_generated(const std::string& test_name) {
        auto expected_filename = test_name_to_expected_file(test_name);

        auto expected = tensor_template + readDataIntoString(expected_filename);
        EXPECT_EQ(get_generated_code(test_name), expected);
    }

//    void assert_generated_defintion(const std::string& input_filename, const std::string& expected_filename, int d = 0) {
//        // parse
//        auto input = readDataIntoString(input_filename);
//        auto expected = readDataIntoString(expected_filename);
//        auto mod = std::make_shared<Module>(parse(input));
//
//        // cleanup
//        auto new_module = apply_default_rewriters(mod);
//
//        // print IR
//        new_module->accept(&printer);
//        // cerr << printer.ast;
//
//        // selective code generation
//        auto def = new_module->decls[d];
//        if (def->is_def()) {
//            def->as_def()->accept(&generator);
//        } else if (def->is_decl()) {
//            auto defs = def->as_decl()->body;
//            for (auto& stmt: defs) {
//                stmt->accept(&generator);
//            }
//        }
//        auto output = oss.str();
//
//        // check output code
//        EXPECT_EQ(output, expected);
//    }

};

TEST_F(GenTest, Definition1) {
    assert_compiles("definition1", get_compiler_path());
}

TEST_F(GenTest, Definition2) {
    assert_compiles("definition2", get_compiler_path());
}

TEST_F(GenTest, Definition3) {
    assert_compiles("definition3", get_compiler_path());
}

TEST_F(GenTest, Definition4) {
    assert_compiles("definition4", get_compiler_path());
}

TEST_F(GenTest, DefinitionCall) {
    assert_compiles("call", get_compiler_path());
}

TEST_F(GenTest, DefinitionCallCondition1) {
    assert_compiles("call_condition1", get_compiler_path());
}

TEST_F(GenTest, DefinitionCallCondition2) {
    assert_compiles("call_condition2", get_compiler_path());
}

//TODO: remove lambda, later
TEST_F(GenTest, DefinitionCallRepeat1) {
    assert_compiles("call_repeat1", get_compiler_path());
}

TEST_F(GenTest, DefinitionCallRepeat2) {
    assert_compiles("call_repeat2", get_compiler_path());
}

TEST_F(GenTest, DefinitionCallRepeat3) {
    assert_compiles("call_repeat3", get_compiler_path());
}

TEST_F(GenTest, DefinitionCallRepeat4) {
    assert_compiles("call_repeat4", get_compiler_path());
}

TEST_F(GenTest, DefinitionCallRepeat5) {
    assert_compiles("call_repeat5", get_compiler_path());
}

TEST_F(GenTest, OuterLoopVar1) {
    assert_compiles("outer_loop_var1", get_compiler_path());
}

TEST_F(GenTest, OuterLoopVar2) {
    assert_compiles("outer_loop_var2", get_compiler_path());
}

TEST_F(GenTest, BFS_Step) {
    assert_compiles("bfs_step", get_compiler_path());
}
