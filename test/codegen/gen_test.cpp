//
// Created by Alexandra Dima on 11/15/21.
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir.h"
#include "einsum_taco/ir/cleanup.h"
#include "einsum_taco/codegen/codegen_visitor.h"
#include "einsum_taco/ir/dump_ast.h"
#include <einsum_taco/parser/heading.h>
#include "../utils.h"

using namespace std;


class GenTest : public testing::Test {
public:
    std::stringstream oss;
    CodeGenVisitor generator;
    std::vector<IRRewriter*> rewriters;
    DumpAstVisitor printer;

    // GenTest() : generator(oss, "test"), rewriters{new IndexVarRewriter(&context)} {}

    GenTest() : generator(oss, "test"),
                rewriters{
                        new TensorVarRewriter(new IRContext()),
                        new FuncDeclRewriter(new IRContext()),
                        new IndexDimensionRewriter(new IRContext())} {}

    void assert_generated(const std::string& input, const std::string& expected) {
        // parse
        auto mod = std::make_shared<Module>(parse(input));

        // cleanup
        auto new_module = apply_rewriters(mod, rewriters);

        // code generation
        new_module->accept(&generator);
        auto output = oss.str();

        // check output code
        EXPECT_EQ(output, expected);
    }

    void assert_generated_defintion(const std::string& input_filename, const std::string& expected_filename, int d = 0) {
        // parse
        auto input = *readFileIntoString(input_filename);
        auto expected = *readFileIntoString(expected_filename);
        auto mod = std::make_shared<Module>(parse(input));

        // cleanup
        auto new_module = apply_rewriters(mod, rewriters);

        // print IR
        new_module->accept(&printer);
        // cerr << printer.ast;

        // selective code generation
        auto def = new_module->decls[d];
        if (def->is_def()) {
            def->as_def()->accept(&generator);
        } else if (def->is_decl()) {
            auto defs = def->as_decl()->body;
            for (auto& stmt: defs) {
                stmt->accept(&generator);
            }
        }
        auto output = oss.str();

        // check output code
        EXPECT_EQ(output, expected);
    }

};

TEST_F(GenTest, Definition1) {
    assert_generated_defintion("codegen/inputs/definition1.txt", "codegen/outputs/definition1.cpp");
}

TEST_F(GenTest, Definition2) {
    assert_generated_defintion("codegen/inputs/definition2.txt", "codegen/outputs/definition2.cpp");
}

TEST_F(GenTest, Definition3) {
    assert_generated_defintion("codegen/inputs/definition3.txt", "codegen/outputs/definition3.cpp");
}

TEST_F(GenTest, Definition4) {
    assert_generated_defintion("codegen/inputs/definition4.txt", "codegen/outputs/definition4.cpp");
}

TEST_F(GenTest, DefinitionCall) {
    assert_generated_defintion("codegen/inputs/call.txt", "codegen/outputs/call.cpp", 1);
}

TEST_F(GenTest, DefinitionCallCondition1) {
    assert_generated_defintion("codegen/inputs/call_condition1.txt", "codegen/outputs/call_condition1.cpp", 1);
}

//TODO: remove lambda, later
TEST_F(GenTest, DefinitionCallRepeat1) {
    assert_generated_defintion("codegen/inputs/call_repeat1.txt", "codegen/outputs/call_repeat1.cpp", 1);
}

TEST_F(GenTest, DefinitionCallRepeat2) {
    assert_generated_defintion("codegen/inputs/call_repeat2.txt", "codegen/outputs/call_repeat2.cpp", 1);
}

TEST_F(GenTest, DefinitionCallRepeat3) {
    assert_generated_defintion("codegen/inputs/call_repeat3.txt", "codegen/outputs/call_repeat3.cpp", 1);
}

TEST_F(GenTest, DefinitionCallRepeat4) {
    assert_generated_defintion("codegen/inputs/call_repeat4.txt", "codegen/outputs/call_repeat4.cpp", 1);
}

TEST_F(GenTest, DefinitionCallRepeat5) {
    assert_generated_defintion("codegen/inputs/call_repeat5.txt", "codegen/outputs/call_repeat5.cpp", 1);
}

TEST_F(GenTest, DefinitionCallCondition2) {
    assert_generated_defintion("codegen/inputs/call_condition2.txt", "codegen/outputs/call_condition2.cpp", 1);
}

TEST_F(GenTest, BFS_Step) {
    assert_generated_defintion("codegen/inputs/bfs_step.txt", "codegen/outputs/bfs_step.cpp");
}
