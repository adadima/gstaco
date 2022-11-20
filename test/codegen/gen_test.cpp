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
#include "einsum_taco/codegen/finch_codegen_visitor.h"
#include "einsum_taco/ir/dump_ast.h"
#include "einsum_taco/gstrt/tensor.h"
#include <einsum_taco/parser/heading.h>
#include "../utils.h"
#include <stdio.h>

using namespace std;

template<typename T>
class BaseGenTest : public testing::TestWithParam<std::tuple<std::string, std::string, bool, T>> {
    virtual void TestBody() = 0;
public:
    std::stringstream oss_cpp;
    std::stringstream oss_h;
    std::stringstream oss_drive;
    FinchCodeGenVisitor generator;
    DumpAstVisitor printer;
    T execution_params;

    BaseGenTest() : generator(&oss_cpp, &oss_h, &oss_drive, std::get<0>(this->GetParam())), execution_params(std::get<3>(this->GetParam())) {}

    static void writeStringToFile(const std::string& filename, const std::string& generated_code) {
        std::ofstream out(filename);
        out << generated_code;
        out.close();
    }

    static std::string get_runtime_dir() {
        return {INCLUDE_GSTACO_RUNTIME};
    }

    static std::string readDataIntoString(const std::string& path) {
        return readFileIntoString(path);
    }

    static std::string get_tensor_template() {
        return readFileIntoString(get_runtime_dir() + "tensor.h");
    }

    std::string tensor_template = get_tensor_template();

    static std::string graph_file_path(const std::string& graph) {
        return get_test_data_dir() + "codegen/graphs/" + graph + ".txt";
    }

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

    static std::string test_name_to_generated_driver(const std::string& test_name) {
        return get_tmp_dir_name() + "driver_" + test_name + ".cpp";
    }

    std::tuple<std::string, std::string, std::string> get_generated_code(const std::string& test_name) {
        // read input
        auto input = readDataIntoString(test_name_to_input_file(test_name));
        std::cout << "INPUT:\n" << input << "\n";
        // parse
        auto mod = std::make_shared<Module>(parse(input));
        auto new_module = apply_default_rewriters(mod);
        std::cout << "PRINT AST\n";
//         print ast for debug
        new_module->accept(&printer);
        std::cout << printer.ast;
        std::cout << new_module->dump();

        // code generation
        new_module->accept(&generator);
        return std::tuple{oss_cpp.str(), oss_h.str(), oss_drive.str()};
    }

    // borrowed from https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
    static std::tuple<int, std::string> exec(const char* cmd) {
        std::array<char, 128> buffer{};
        std::string result;

        // start process to run cmd
        FILE* pipe = popen(cmd, "r");
        if (!pipe) {
            DEBUG_LOG << "Couldn't open process." << std::endl;
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
        if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
        {
            if( errno == EEXIST ) {
                // alredy exists
            } else {
                // something else
                std::cout << "cannot create directory: " << strerror(errno) << std::endl;
                throw std::runtime_error( strerror(errno) );
            }
        }
    }

    void cleanup(const std::string& in, const std::string& out, const std::string& header, const std::string& driver, bool add_main) {
        if (add_main) {
            remove(in.c_str());
            remove(out.c_str());
            remove(header.c_str());
            remove(driver.c_str());
        }
    }

    void assert_compiles(const std::string& test_name, const std::string& compiler_path, bool add_main = true) {
        create_dir(get_tmp_dir_name());

        std::string tmp_header = test_name_to_header(test_name);
        std::string tmp_in = test_name_to_tmp_input(test_name);
        std::string tmp_out = test_name_to_tmp_output(test_name);
        std::string driver;
        if (add_main) {
            driver = test_name_to_generated_driver(test_name);
        } else {
            driver = test_name_to_driver(test_name);
        }

        // generate code from input file and add dummy main function
        std::string code;
        std::string header;
        std::string drv;

        std::tie(code, header, drv) = get_generated_code(test_name);

        // write generated code to temporary cpp file
        writeStringToFile(tmp_header, header);

        // write generated code to temporary cpp file
        writeStringToFile(tmp_in, code);


        if (add_main) {
            writeStringToFile(driver, drv);
        }

        std::stringstream cmdss;
        // compile the temporary cpp file with the given compiler
        cmdss << compiler_path << " -o " << tmp_out << " -std=c++17 ";
        GTEST_LOG_(INFO) << get_julia_include_dir() << "\n";
        cmdss << " -I'" << get_julia_include_dir() << "' -fPIC -I" << get_finch_embed_dir() << "  -L'" << get_julia_lib_dir() << "' -L" << get_finch_embed_dir() << " -Wl,-rpath,'" << get_julia_lib_dir() << "' -ljulia -lfinch";
#if __APPLE__
        cmdss << " -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/";
#endif
        cmdss << " " << tmp_in << " " << driver;
        std::string cmd = cmdss.str();
        GTEST_LOG_(INFO) << cmd << "\n";

        int status_code;
        std::string output;
        std::tie(status_code, output) = exec(cmd.c_str());

        // remove generated files
//        cleanup(tmp_in, tmp_out, tmp_header, driver, add_main);

        // check compilation process finished successfully
        EXPECT_EQ(status_code, 0);
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

    void check_page_rank_output(const std::string& output, const std::vector<float>& ranks) {
        auto lines = getLines(output);

        for (int i=0; i < lines.size(); i++) {
            auto result = std::stof(lines[i]);
            auto expected = ranks[i];
            EXPECT_FLOAT_EQ(result, expected);
        }
    }

    using inner_checker=void (*)(std::string);

    template<typename R>
    using outer_checker=inner_checker (*)(R);

    template<typename callable>
    void assert_runs(const std::string& test_name, const std::string& graph_name, std::vector<std::string> args, callable checker) {
        std::string cmd = test_name_to_tmp_output(test_name) + " " + graph_file_path(graph_name);
        for (auto &arg: args) {
            cmd += " " + arg;
        }

        GTEST_LOG_(INFO) << cmd << "\n";

        int status_code;
        std::string output;
        std::tie(status_code, output) = exec(cmd.c_str());

        // check run process finished successfully
        EXPECT_EQ(status_code, 0);

        // check the computed ranks are correct
        checker(output);
    }
};

struct ExecutionParams {
    std::string output_filename;

    explicit ExecutionParams(std::string output_filename="out") : output_filename(get_test_dir() + "tmp/codegen/" + output_filename) {}
};

class GenTest : public BaseGenTest<ExecutionParams>{};

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
                make_tuple("definition2", get_compiler_path(), true, ExecutionParams()),
                make_tuple("definition1", get_compiler_path(), true, ExecutionParams()),
                make_tuple("definition3", get_compiler_path(), true, ExecutionParams()),
                make_tuple("definition4", get_compiler_path(), true, ExecutionParams()),
                make_tuple("call", get_compiler_path(), true, ExecutionParams()),
                make_tuple("call_condition1", get_compiler_path(), true, ExecutionParams()),
//                make_tuple("call_condition2", get_compiler_path(), true, ExecutionParams()),
                make_tuple("call_repeat1", get_compiler_path(), true, ExecutionParams()),
                make_tuple("call_repeat2", get_compiler_path(), true, ExecutionParams()),
                make_tuple("call_repeat3", get_compiler_path(), true, ExecutionParams()),
//                make_tuple("call_repeat4", get_compiler_path(), true, ExecutionParams()),
//                make_tuple("call_repeat5", get_compiler_path(), true, ExecutionParams())
                make_tuple("outer_loop_var1", get_compiler_path(), true, ExecutionParams()),
                make_tuple("outer_loop_var2", get_compiler_path(), true, ExecutionParams())
        ));

struct PageRankExecutionParams : ExecutionParams {
    std::string graph_name;
    std::vector<float> expected_ranks;

    PageRankExecutionParams(std::string  graph_name, std::vector<float>  expected_ranks) :
            ExecutionParams("PR_" + graph_name + ".txt"), graph_name(std::move(graph_name)), expected_ranks(std::move(expected_ranks)) {}
};

class PageRankTest : public BaseGenTest<PageRankExecutionParams> {};

TEST_P(PageRankTest, PageRank) {
    auto test_name = std::get<0>(GetParam());
    auto compiler = std::get<1>(GetParam());
    auto add_main = std::get<2>(GetParam());
    assert_compiles(test_name, compiler, add_main);

    auto graph = std::get<3>(GetParam()).graph_name;
    auto ranks = std::get<3>(GetParam()).expected_ranks;
    auto out = std::get<3>(GetParam()).output_filename;
    assert_runs(test_name, graph, {"0.85", out}, [&](std::string output){ check_page_rank_output(output, ranks);});
}

INSTANTIATE_TEST_CASE_P(
        PageRankTestSuite,
        PageRankTest,
        ::testing::Values(
                make_tuple("pagerank", get_compiler_path(), false, PageRankExecutionParams(
                        "graph1",
                        {0.372532, 0.195814, 0.394155, 0.0375})),
                make_tuple("pagerank", get_compiler_path(), false, PageRankExecutionParams(
                        "graph2",
                        {0.432749, 0.233918, 0.333333}))
        ));


struct BFSExecutionParams : ExecutionParams {
    std::string graph_name;
    int source;
    std::vector<int> expected_dist;

    BFSExecutionParams(std::string  graph_name, int source, std::vector<int>  expected_dist) :
            ExecutionParams("BFS_" + graph_name + ".txt"),
            graph_name(graph_name), source(source), expected_dist(expected_dist) {}
};

class BFSTest : public BaseGenTest<BFSExecutionParams> {
public:
    void check_bfs_output(const std::string& output, const std::vector<int>& distances) {
        auto lines = getLines(output);

        for (int i=0; i < lines.size(); i++) {
            std:cerr << lines[i] << std::endl;
            auto result = std::stoi(lines[i]);
            auto expected = distances[i];
            EXPECT_EQ(result, expected);
        }
    }
};

TEST_P(BFSTest, BFS) {
    auto test_name = std::get<0>(GetParam());
    auto compiler = std::get<1>(GetParam());
    auto add_main = std::get<2>(GetParam());
    assert_compiles(test_name, compiler, add_main);

    auto graph = std::get<3>(GetParam()).graph_name;
    auto dist = std::get<3>(GetParam()).expected_dist;
    auto source = std::get<3>(GetParam()).source;
    auto out = std::get<3>(GetParam()).output_filename;
    assert_runs(test_name, graph, {std::to_string(source), out}, [&](std::string output) { check_bfs_output(output, dist);});
}

INSTANTIATE_TEST_CASE_P(
        BFSTestSuite,
        BFSTest,
        ::testing::Values(
                make_tuple("bfs", get_compiler_path(), false, BFSExecutionParams(
                        "graph1", 0,
                        {-2, 1, 1, -1})),
                make_tuple("bfs", get_compiler_path(), false, BFSExecutionParams(
                        "graph2", 0,
                        {-2, 1, 1})),
                make_tuple("bfs", get_compiler_path(), false, BFSExecutionParams(
                        "graph3", 0,
                        {-2, -1, -1, -1, -1})),
                make_tuple("bfs", get_compiler_path(), false, BFSExecutionParams(
                        "graph3", 4,
                        {5, 3, 4, 5, -2})),
                make_tuple("bfs", get_compiler_path(), false, BFSExecutionParams(
                        "graph3", 1,
                        {2, -2, -1, -1, -1})),
                make_tuple("bfs", get_compiler_path(), false, BFSExecutionParams(
                        "graph3", 2,
                        {2, 3, -2, -1, -1})),
                make_tuple("bfs", get_compiler_path(), false, BFSExecutionParams(
                        "graph3", 3,
                        {2, 3, 4, -2, -1}))
        ));

struct SSSPExecutionParams : ExecutionParams {
    std::string graph_name;
    int source;
    int infinity;
    std::vector<float> expected_dist;

    SSSPExecutionParams(std::string  graph_name, int source, int infinity, std::vector<float>  expected_dist) :
            ExecutionParams("SSSP_" + graph_name + ".txt"),
    graph_name(graph_name), source(source), infinity(infinity), expected_dist(expected_dist) {}
};

class SSSPTest : public BaseGenTest<SSSPExecutionParams> {
public:
    void check_sssp_output(const std::string& output, const std::vector<float>& distances) {
        auto lines = getLines(output);

        for (int i=0; i < lines.size(); i++) {
            std:cerr << lines[i] << std::endl;
            auto result = std::stof(lines[i]);
            auto expected = distances[i];
            EXPECT_FLOAT_EQ(result, expected);
        }
    }
};

TEST_P(SSSPTest, SSSP) {
    auto test_name = std::get<0>(GetParam());
    auto compiler = std::get<1>(GetParam());
    auto add_main = std::get<2>(GetParam());
    assert_compiles(test_name, compiler, add_main);

    auto graph = std::get<3>(GetParam()).graph_name;
    auto dist = std::get<3>(GetParam()).expected_dist;
    auto source = std::get<3>(GetParam()).source;
    auto inf = std::get<3>(GetParam()).infinity;
    auto out = std::get<3>(GetParam()).output_filename;
    assert_runs(test_name, graph, {std::to_string(source), std::to_string(inf), out}, [&](std::string output) { check_sssp_output(output, dist);});
}

INSTANTIATE_TEST_CASE_P(
        SSSPTestSuite,
        SSSPTest,
        ::testing::Values(
                make_tuple("sssp", get_compiler_path(), false, SSSPExecutionParams(
                        "graph4", 0, 1000,
                        {0, 1.0, 1.7, 1000.0})),
                make_tuple("sssp", get_compiler_path(), false, SSSPExecutionParams(
                        "graph5", 0, 1000,
                        {0, 2.0, 5.0, 1.0})),
                make_tuple("sssp", get_compiler_path(), false, SSSPExecutionParams(
                        "graph3", 4, 1000,
                        {4.0, 3.0, 2.0, 1.0, 0})),
                make_tuple("sssp", get_compiler_path(), false, SSSPExecutionParams(
                        "graph2", 0, 1000,
                        {0, 1.0, 2.0})),
                make_tuple("sssp", get_compiler_path(), false, SSSPExecutionParams(
                        "graph2", 1, 1000,
                        {2.0, 0, 1.0})),
                make_tuple("sssp", get_compiler_path(), false, SSSPExecutionParams(
                        "graph2", 2, 1000,
                        {1.0, 2.0, 0})),
                make_tuple("sssp", get_compiler_path(), false, SSSPExecutionParams(
                        "graph1", 0, 1000,
                        {0, 1.0, 2.0, 1000.0}))
        ));

struct BCExecutionParams : ExecutionParams {
    std::string graph_name;
    int source;
    std::vector<float> expected;

    BCExecutionParams(const std::string&  graph_name, int source, const std::vector<float>&  expected) :
            ExecutionParams("BC_" + graph_name + ".txt"),
            graph_name(graph_name), source(source), expected(expected) {}
};

class BCTest : public BaseGenTest<BCExecutionParams> {
public:
    void check_bc_output(const std::string& output, const std::vector<float>& distances) {
        std::cout << "Output: " << output << "\n";
        auto lines = getLines(output);

        for (int i=0; i < lines.size(); i++) {
            DEBUG_LOG << lines[i] << std::endl;
            auto result = std::stof(lines[i]);
            auto expected = distances[i];
            EXPECT_EQ(result, expected);
        }
    }
};

TEST_P(BCTest, BC) {
    auto test_name = std::get<0>(GetParam());
    auto compiler = std::get<1>(GetParam());
    auto add_main = std::get<2>(GetParam());
    assert_compiles(test_name, compiler, add_main);

    auto graph = std::get<3>(GetParam()).graph_name;
    auto dist = std::get<3>(GetParam()).expected;
    auto source = std::get<3>(GetParam()).source;
    auto out = std::get<3>(GetParam()).output_filename;
    assert_runs(test_name, graph, {std::to_string(source), out}, [&](std::string output) { check_bc_output(output, dist);});
}

INSTANTIATE_TEST_CASE_P(
        BCTestSuite,
        BCTest,
        ::testing::Values(
//                make_tuple("bc", get_compiler_path(), false, BCExecutionParams(
//                        "graph1", 0,
//                        {0, 0, 0, 0})),
                make_tuple("bc", get_compiler_path(), false, BCExecutionParams(
                        "graph3", 4,
                        {0, 0, 1, 2, 0}))
        ));