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

    void assert_generated_defintion(const std::string& input_filename, const std::string& expected_filename, int d = 0) {
        auto input = *readFileIntoString(input_filename);
        auto expected = *readFileIntoString(expected_filename);
        auto mod = parse(input);
        mod.accept(&rewriter);
//        rewriter.module->accept(&printer);
//        cout << printer.ast;
        auto new_module = rewriter.module;
//        new_module->accept(&printer);
//        cerr << printer.ast;
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

TEST_F(GenTest, DefinitionCallStar) {
    assert_generated_defintion("codegen/inputs/call_condition1.txt", "codegen/outputs/call_condition1.cpp", 1);
}
