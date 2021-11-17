//
// Created by Alexandra Dima on 11/15/21.
#include <iostream>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir.h"

class GenTest : public testing::Test {
protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).

    }
};

TEST_F(GenTest, FailingTest ) {
    EXPECT_EQ ("1+2",  "4");
}
