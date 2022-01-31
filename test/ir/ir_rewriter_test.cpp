//
// Created by Alexandra Dima on 10.01.2022.
//

#include <iostream>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/ir/dump_ast.h"
#include "einsum_taco/ir/cleanup.h"

#include <einsum_taco/parser/heading.h>
#include "../utils.h"

using namespace std;

class Cleanup : public testing::Test {
public:
    DumpAstVisitor* printer;
    std::vector<IRRewriter*> rewriters;

    Cleanup() :
        printer(new DumpAstVisitor()),
        rewriters{
                new TensorVarRewriter(new IRContext()),
                new FuncDeclRewriter(new IRContext()),
                new IndexDimensionRewriter(new IRContext())} {}
};

//TODO: write more visitor tests!

TEST_F(Cleanup, IndexVars1) {
    auto input = *readFileIntoString("rewriter/inputs/index_vars1.txt");
    auto mod = std::make_shared<Module>(parse(input));
    auto module = apply_rewriters(mod, rewriters);
    EXPECT_EQ(module->dump(), input);

    auto ast = *readFileIntoString("rewriter/outputs/index_vars1.txt");
    module->accept(printer);
    EXPECT_EQ(printer->ast, ast);
}
