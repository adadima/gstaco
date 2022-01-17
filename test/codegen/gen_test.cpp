//
// Created by Alexandra Dima on 11/15/21.
#include <iostream>
#include <sstream>
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

    void assert_generated_defintion(const std::string& input, const std::string& expected) {
        auto mod = parse(input);
        mod.accept(&rewriter);
        rewriter.module->accept(&printer);
//        cout << printer.ast;
        auto new_module = rewriter.module;
        auto def = new_module->decls[0];
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
    assert_generated_defintion(in, expected);
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
