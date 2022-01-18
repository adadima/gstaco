//
// Created by Alexandra Dima on 11/15/21.
#include <iostream>
#include <sstream>
#include <fstream>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir.h"
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/codegen/codegen_visitor.h"
#include "einsum_taco/ir/dump_ast.h"
#include <einsum_taco/parser/heading.h>
#include "../utils.h"

using namespace std;

class Cleanup : public IRRewriter, public testing::Test {
public:
    DumpAstVisitor* printer;
    Cleanup() : IRRewriter(new IRContext()) {
        printer = new DumpAstVisitor();
    }

    void check_definition(const shared_ptr<Definition>& def, const map<string, shared_ptr<Expression>>& expected) {
        for (auto &[key, value] : def->getIndexVarDims(context)) {

        }
    }
};

class GenTest : public testing::Test {
public:
    std::stringstream oss;
    IRContext context;
    CodeGenVisitor generator;
    IRRewriter rewriter;
    DumpAstVisitor printer;

    GenTest() : generator(oss, "test"), rewriter(&context) {}

    void assert_generated(const std::string& input, const std::string& expected) {
        auto mod = parse(input);
        mod.accept(&rewriter);

        auto new_module = rewriter.module;
        new_module->accept(&generator);

        auto output = oss.str();
        EXPECT_EQ(output, expected);
    }

    static std::string readFileIntoString(const string& path) {
        std::ifstream input_file(path);
        if (!input_file.is_open()) {
            cerr << "Could not open the file - '"
                 << path << "'" << endl;
            exit(EXIT_FAILURE);
        }
        return string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
    }

    void assert_generated_defintion(const std::string& input_filename, const std::string& expected_filename, int d = 0) {
        auto input = input_filename; // readFileIntoString(input_filename);
        auto expected = expected_filename; // readFileIntoString(expected_filename);
        auto mod = parse(input);
        mod.accept(&rewriter);
//        rewriter.module->accept(&printer);
//        cout << printer.ast;
        auto new_module = rewriter.module;
//        new_module->accept(&printer);
//        cout << printer.ast;
        auto def = new_module->decls[d];
        if (def->is_def()) {
            def->as_def().accept(&generator);
        } else if (def->is_decl()) {
            def->as_decl().body[0]->accept(&generator);
        }

        auto output = oss.str();
        EXPECT_EQ(output, expected);
    }

};

TEST_F(GenTest, Definition1) {
    auto in = R"(Let func(B int[10][20], C int[20]) -> (A int[10])
    A[i] = B[i][j] * C[j] | j:(+, 0)
End)";
    auto expected = R"(for(int i=0; i<10; i++) {
    auto init_j = 0;
    for(int j=0; j<20; j++) {
        init_j = init_j + B[i][j] * C[j];
    }
    A[i] = init_j;
}
)";
    auto p = std::filesystem::current_path().string();
    assert_generated_defintion(p + "/inputs/definition1.txt", p + "/outputs/definition1.cpp");
}

TEST_F(GenTest, Definition2) {
    auto in = R"(Let func(B float[N][10][M][20], C float[10], D float[20]) -> (A float[N][M])
    A[i][k] = B[i][j][k][l] * C[j] * D[l] | j:(+, 0), l:(+, 0)
End)";
    auto expected = R"(for(int i=0; i<N; i++) {
    for(int k=0; k<M; k++) {
        auto init_j = 0;
        for(int j=0; j<10; j++) {
            auto init_l = 0;
            for(int l=0; l<20; l++) {
                init_l = init_l + B[i][j][k][l] * C[j] * D[l];
            }
            init_j = init_j + init_l;
        }
        A[i][k] = init_j;
    }
}
)";
    assert_generated_defintion(in, expected);
}

TEST_F(GenTest, Definition3) {
    auto in = R"(Let func(edges int[N][M], frontier_list int[10][M], visited int[N], round int) -> (frontier int[N])
    frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(OR, 0)
End)";
    auto expected = R"(for(int j=0; j<N; j++) {
    auto init_k = 0;
    for(int k=0; k<M; k++) {
        init_k = init_k || edges[j][k] * frontier_list[round][k] * (visited[j] == 0);
    }
    frontier[j] = init_k;
}
)";
    assert_generated_defintion(in, expected);
}

TEST_F(GenTest, Definition4) {
    auto in = R"(Let func(B int[10][20], C int[20], j int, i int, k int) -> (A int[10])
    A[i] = B[i][j] * C[j] * C[k] | j:(+, 0)
End)";
    auto expected = R"(for(int i=0; i<10; i++) {
    auto init_j = 0;
    for(int j=0; j<20; j++) {
        init_j = init_j + B[i][j] * C[j] * C[k];
    }
    A[i] = init_j;
}
)";
    assert_generated_defintion(in, expected);
}

TEST_F(GenTest, DefinitionCall) {
    auto in = R"(Let fib(A int, B float) -> (C int, D float)

End

Let func(C int, D float) -> (A int, B float)
    A, B = fib(C, D)
End)";
    auto expected = R"(auto out = fib(C, D);
A = std::get<0>(out);
B = std::get<1>(out);
)";
    assert_generated_defintion(in, expected, 1);
}

TEST_F(GenTest, DefinitionCallRepeat1) {
    auto in = R"(Let fib(A int, B float) -> (C int, D float)

End

Let func(C int, D float) -> (A int, B float)
    A, B = fib*(C, D) | 100
End)";
    auto expected = R"(auto out = ([&]{
auto out = fib(C, D);
auto& [out0, out1] = out;
for(int iter=0; iter<99; iter++) {
    std::tie(out0, out1) = fib(out0, out1);
}
return std::tuple<int, float>{out0, out1};
}());
A = std::get<0>(out);
B = std::get<1>(out);
)";
    assert_generated_defintion(in, expected, 1);
}


TEST_F(GenTest, DefinitionCallRepeat2) {
    auto in = R"(Let fib(A int, B int) -> (C int, D int)
    C = A
    D = B
End

Let func() -> (A int[N], B int[N])
    A[i], B[i] = fib*(1, 2) | 3
End)";
    auto expected = R"(for(int i=0; i<N; i++) {
    auto out = ([&]{
auto out = fib(1, 2);
auto& [out0, out1] = out;
    for(int iter=0; iter<2; iter++) {
        std::tie(out0, out1) = fib(out0, out1);
    }
return std::tuple<int, int>{out0, out1};
}());
    A[i] = std::get<0>(out);
    B[i] = std::get<1>(out);
}
)";
    assert_generated_defintion(in, expected, 1);
}

TEST_F(GenTest, DefinitionCallRepeat3) {
    auto in = R"(Let fib(A int, B int) -> (C int, D int)
    C = A
    D = B
End

Let func() -> (A int[N], B int[N][M])
    A[i], B[i][j] = fib*(1, 2) | 3
End)";
    auto expected = R"(for(int i=0; i<N; i++) {
    for(int j=0; j<M; j++) {
        auto out = ([&]{
auto out = fib(1, 2);
auto& [out0, out1] = out;
for(int iter=0; iter<2; iter++) {
    std::tie(out0, out1) = fib(out0, out1);
}
return std::tuple<int, int>{out0, out1};
}());
        A[i] = std::get<0>(out);
        B[i][j] = std::get<1>(out);
    }
}
)";
    assert_generated_defintion(in, expected, 1);
}

TEST_F(GenTest, DefinitionCallStar) {
    auto in = R"(Let fib(A int, B float) -> (C int, D float)

End

Let func(C int, D float) -> (A int, B float)
    A, B = fib*(C, D) | (A == 0)
End)";
    auto expected = R"(auto out = fib(C, D);
auto& [out0, out1] = out;
while(!(A == 0)) {
    auto out = ([&]{
auto out = fib(C, D);
auto& [out0, out1] = out;
for(int iter=0; iter<99; iter++) {
    std::tie(out0, out1) = fib(out0, out1);
}
return std::tuple<int, float>{out0, out1};
}());
}
A = std::get<0>(out);
B = std::get<1>(out);
)";
    assert_generated_defintion(in, expected, 1);
}
