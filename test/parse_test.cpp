//
// Created by Alexandra Dima on 19.12.2021.
//
#include <iostream>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir.h"
#include <einsum_taco/parser/heading.h>
#include "utils.h"

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

TEST(ParseTest, Bcentrality) {
    auto expected = R"(
Let Init(source int) -> (frontier_list int[N][N], num_paths int[N], deps int[N], visited int[N])
    num_paths[j] = j == source
    deps[j] = 0
    visited[j] = j == source
    frontier_list[r][j] = r == 0 && j == source
End

Let Frontier(frontier_list int[N][N], visited int[N], round int) -> (frontier int[N])
    frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(OR, 0)
End

Let Forward_Step(frontier_list int[N][N], num_paths int[N], visited int[N], round int) -> (forward_frontier_list int[N][N], forward_num_paths int[N], forward_visited int[N], forward_round int)
    frontier = Frontier(frontier_list, visited, round - 1)
    forward_frontier_list[r][j] = frontier[j] * (r == round) + frontier_list[r][j] * (r != round)
    forward_num_paths[j] = edges[j][k] * frontier_list[round - 1][k] * (visited[j] == 0) * num_paths[k] | k:(+, num_paths[j])
    forward_visited[j] = edges[j][k] * frontier_list[round - 1][k] * (visited[j] == 0) | k:(OR, visited[j])
    forward_round = round + 1
End

Let Forward(frontier_list int[N][N], num_paths int[N], visited int[N]) -> (new_forward_frontier_list int[N][N], new_forward_num_paths int[N], new_forward_visited int[N], new_forward_round int)
    new_forward_frontier_list, new_forward_num_paths, new_forward_visited, new_forward_round = Forward_Step*(frontier_list, num_paths, visited, 1) | (#1[#4 - 1] == 0)
End

Let Backwards_Vertex(frontier_list int[N][N], num_paths int[N], deps int[N], visited int[N], round int) -> (backward_deps int[N], backward_visited int[N])
    backward_deps[j] = deps[j] + num_paths[j] * frontier_list[round][j]
    backward_visited[j] = frontier_list[round][j]
End

Let Backwards_Edge(frontier_list int[N][N], num_paths int[N], deps int[N], visited int[N], round int) -> (backward_deps int[N], backward_round int)
    backward_deps[j] = edges[k][j] * frontier_list[round][k] * visited[k] * deps[k] | k:(+, deps[j])
    backward_round = round - 1
End

Let Backward_Step(frontier_list int[N][N], num_paths int[N], deps int[N], visited int[N], round int) -> (final_frontier_list int[N][N], final_num_paths int[N], final_deps int[N], final_visited int[N], final_round int)
    final_frontier_list[r][j] = frontier_list[r][j]
    final_num_paths[j] = num_paths[j]
    backward_deps, final_visited = Backwards_Vertex(frontier_list, num_paths, deps, visited, round)
    final_deps, final_round = Backwards_Edge(frontier_list, num_paths, backward_deps, final_visited, round)
End

Let BC() -> (final_deps int[N])
    frontier_list, num_paths, deps, visited = Init(source)
    forward_frontier_list, forward_num_paths, _, forward_round = Forward(frontier_list, num_paths, visited)
    new_deps, new_visited = Backwards_Vertex(forward_frontier_list, forward_num_paths, deps, visited, forward_round)
    _, _, final_deps, _, _ = Backward_Step*(forward_frontier_list, forward_num_paths, new_deps, new_visited, forward_round) | (#5 == 0)
End
)";
    auto f = fopen("../../apps/bcentrality.txt", "r");
    auto module = parse_module(f);
    EXPECT_EQ(module.dump(), expected);
}