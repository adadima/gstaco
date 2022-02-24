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

    Cleanup() :
        printer(new DumpAstVisitor()) {}
};

//TODO: write more visitor tests!

TEST_F(Cleanup, IndexVars1) {
    auto input = readFileIntoString(get_test_data_dir() + "rewriter/inputs/index_vars1.txt");
    auto mod = std::make_shared<Module>(parse(input));
    auto module = apply_default_rewriters(mod);
    EXPECT_EQ(module->dump(), input);

    auto ast = readFileIntoString(get_test_data_dir() + "rewriter/outputs/index_vars1.txt");
    module->accept(printer);
    EXPECT_EQ(printer->ast, ast);
}
