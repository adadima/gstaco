//
// Created by Alexandra Dima on 19.12.2021.
//
#include <iostream>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir.h"
#include <einsum_taco/parser/heading.h>
#include "../utils.h"

TEST(ParseTest, LiteralsTest) {
    EXPECT_EQ (parse("6").dump(),  "\n6\n");

    EXPECT_EQ (parse("true").dump(),  "\ntrue\n");

    EXPECT_EQ (parse("6.0").dump().rfind("\n6.0", 0),  0);
}


TEST(ParseTest, ReadAccessTest1) {
    EXPECT_EQ(parse("x = A").dump(), "\nx = A\n");
}

TEST(ParseTest, ReadAccessTest2) {
    EXPECT_EQ(parse("x = A[j]").dump(), "\nx = A[j]\n");
}


TEST(ParseTest, ReadAccessTest3) {
    EXPECT_EQ(parse("x = A[j][k][j + k]").dump(), "\nx = A[j][k][j + k]\n");
}

TEST(ParseTest, BinaryExprTest) {
    EXPECT_EQ(parse("j + k").dump(), "\nj + k\n");

    EXPECT_EQ(parse("j - k").dump(), "\nj - k\n");

    EXPECT_EQ(parse("(j + k) * (j - k)").dump(), "\n(j + k) * (j - k)\n");

    EXPECT_EQ(parse("(j + k) * (j - k) / 2").dump(), "\n(j + k) * (j - k) / 2\n");

    EXPECT_EQ(parse("2 - (j - k)").dump(), "\n2 - (j - k)\n");

    EXPECT_EQ(parse("2 / ((j + k) * (j - k) / 2)").dump(), "\n2 / ((j + k) * (j - k) / 2)\n");

    EXPECT_EQ(parse("(j + k) % 2").dump(), "\n(j + k) % 2\n");

    EXPECT_EQ(parse("j == k").dump(), "\nj == k\n");

    EXPECT_EQ(parse("j + k != k").dump(), "\nj + k != k\n");

    EXPECT_EQ(parse("j + k > j - k").dump(), "\nj + k > j - k\n");

    EXPECT_EQ(parse("x = A[j][k][j + k] < A[0][2][j]").dump(), "\nx = A[j][k][j + k] < A[0][2][j]\n");

    EXPECT_EQ(parse("j + k > j - k || true").dump(), "\nj + k > j - k || true\n");

    EXPECT_EQ(parse("(j + k > j - k || true) && false").dump(), "\n(j + k > j - k || true) && false\n");

    //auto neg = einsum::IR::make<einsum::NotExpression>(bool1);
}

TEST(ParseTest, DefinitionTest) {
    const std::string def = "frontier[j] = edges[j][k] * frontier_list[2][k] * (visited[j] == 0) | k:(OR, 0)";
    EXPECT_EQ(parse(def).dump(),"\n" + def + "\n");
}


TEST(ParseTest, FuncDeclTest1) {
    const std::string func =R"(
Let Round(round_in int) -> (round_out int)
    round_out = round_in * 2
End)";
    EXPECT_EQ(parse(func).dump(),func + "\n");
}

TEST(ParseTest, FuncDeclTest2) {
    const std::string func = R"(
Let Frontier(frontier_list int[N][N], visited int[N], round_in int) -> (frontier int[N], round_out int)
    frontier[j] = edges[j][k] * frontier_list[2][k] * (visited[j] == 0) | k:(OR, 0)
    round_out = round_in * 2
End)";
    EXPECT_EQ(parse(func).dump(), func + "\n");
}

// Round(0)
TEST(ParseTest, CallTest) {
    EXPECT_EQ(parse("Round(0)").dump(), "\nRound(0)\n");
}


// Round(0)
TEST(ParseTest, CallMultipleInputsTest) {

    EXPECT_EQ(parse("Round(0, A)").dump(), "\nRound(0, A)\n");

    EXPECT_EQ(parse("round_out, unused_out = Round(0, A)").dump(), "\nround_out, unused_out = Round(0, A)\n");
}

// Round*(A) | 3
TEST(ParseTest, CallStarRepeatTest) {
    EXPECT_EQ(parse("Round*(A) | (3)").dump(), "\nRound*(A) | (3)\n");

    EXPECT_EQ(parse("round_out = Round*(A) | (3)").dump(), "\nround_out = Round*(A) | (3)\n");
}


// Round*(A) | (#1 == 2)
TEST(ParseTest, CallStarConditionTest) {

    EXPECT_EQ(parse("Round*(0, A) | (#1 == 2)").dump(), "\nRound*(0, A) | (#1 == 2)\n");

    EXPECT_EQ(parse("round_out, unused_out = Round*(0, A) | (#1 == 2)").dump(), "\nround_out, unused_out = Round*(0, A) | (#1 == 2)\n");
}

TEST(ParseTest, Formats) {
    const std::string expected = R"(
Let Main(edges int[N][SparseList[M]]) -> (A int)
End)";

    EXPECT_EQ(parse(R"(
Let Main(edges int[Dense[N]][SparseList[M]]) -> (A int)

End)").dump(), expected + "\n");

    EXPECT_EQ(parse(R"(
Let Main(edges int[N][SparseList[M]]) -> (A int)

End)").dump(), expected + "\n");
}

TEST(ParseTest, CSRTensorVar) {
    const std::string func = R"(edges int[N][SparseList[N]])";
    EXPECT_EQ(parse(func).dump(), "\nedges int[N][SparseList[N]]\n");

    EXPECT_EQ(parse(R"(edges int[N][M])").dump(), "\nedges int[N][M]\n");
}

TEST(ParseTest, Bcentrality) {
    auto expected = readFileIntoString(get_test_data_dir() + "parser/inputs/bcentrality.txt");
    auto module = parse(expected);
    EXPECT_EQ(module.dump(), expected);
}