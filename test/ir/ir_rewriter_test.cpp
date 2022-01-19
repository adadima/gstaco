//
// Created by Alexandra Dima on 10.01.2022.
//

#include <iostream>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir_rewriter.h"
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

//TODO: write more visitor tests!

TEST_F(Cleanup, IndexVars1) {
    auto input = *readFileIntoString("rewriter/inputs/index_vars1.txt");
    auto mod = parse(input);

    mod.accept(this);
    EXPECT_EQ(module->dump(), input);

    auto ast = *readFileIntoString("rewriter/outputs/index_vars1.txt");
    module->accept(printer);
    EXPECT_EQ(printer->ast, ast);
}
