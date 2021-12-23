//
// Created by Alexandra Dima on 11/15/21.
#include <iostream>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir.h"
#include <initializer_list>

typedef struct yy_buffer_state * YY_BUFFER_STATE;

class DumpTest : public testing::Test {
public:
    static inline const auto zero = einsum::IR::make<einsum::Literal>(0, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Int));
    static inline const auto two = einsum::IR::make<einsum::Literal>(2, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Int));
    static inline const auto yes = einsum::IR::make<einsum::Literal>(true, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Bool));
    static inline const auto no = einsum::IR::make<einsum::Literal>(false, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Bool));

    static inline const auto add = einsum::Type::make<einsum::AddOp>();
    static inline const auto sub = einsum::Type::make<einsum::SubOp>();
    static inline const auto mul = einsum::Type::make<einsum::MulOp>();
    static inline const auto div = einsum::Type::make<einsum::DivOp>();
    static inline const auto and_ = einsum::Type::make<einsum::AndOp>();
    static inline const auto or_ = einsum::Type::make<einsum::OrOp>();
    static inline const auto not_ = einsum::Type::make<einsum::NotOp>();
    static inline const auto lt = einsum::Type::make<einsum::LtOp>();
    static inline const auto lte = einsum::Type::make<einsum::LteOp>();
    static inline const auto gt = einsum::Type::make<einsum::GtOp>();
    static inline const auto gte = einsum::Type::make<einsum::GteOp>();
    static inline const auto eq = einsum::Type::make<einsum::EqOp>();
    static inline const auto neq = einsum::Type::make<einsum::NeqOp>();

    static inline const auto i = einsum::IR::make<einsum::IndexVar>("i", 0);
    static inline const auto j = einsum::IR::make<einsum::IndexVar>("j", 0);
    static inline const auto k = einsum::IR::make<einsum::IndexVar>("k", 0);
    static inline const auto jExpr = einsum::IR::make<einsum::IndexVarExpr>(j);
    static inline const auto kExpr = einsum::IR::make<einsum::IndexVarExpr>(k);
    static inline const auto jkExpr = einsum::IR::make<einsum::ArithmeticExpression>(jExpr, kExpr, std::make_shared<einsum::AddOp>());
    static inline const auto intType = einsum::Type::make<einsum::Datatype>(
                                                einsum::Datatype::Kind::Int
                                        );
    static inline const auto varType = einsum::Type::make<einsum::TensorType>();

    template <typename V>
    static std::shared_ptr<einsum::TensorVar> make_tensor(std::string name, std::initializer_list<std::shared_ptr<einsum::Expression>> dimensions) {

        auto tType = einsum::IR::make<einsum::TensorType>(
                        einsum::Datatype::make_datatype<V>(),
                        dimensions
                    );
        return einsum::IR::make<einsum::TensorVar>(name, tType);
    }

    static inline const auto A = make_tensor<int>("A",
                {einsum::IR::make<einsum::Literal>(2, intType),
                einsum::Type::make<einsum::Literal>(3, intType),
                einsum::Type::make<einsum::Literal>(1, intType)}
        );

    std::shared_ptr<einsum::Definition> definition1() {
        std::initializer_list<std::shared_ptr<einsum::Expression>> dims = {einsum::IR::make<einsum::Literal>(2, intType),
                                                                           einsum::Type::make<einsum::Literal>(3, intType),
                                                                           einsum::Type::make<einsum::Literal>(1, intType)};
        auto frontier = make_tensor<int>("frontier", dims);
        auto visited = make_tensor<int>("visited", dims);
        auto edges = make_tensor<int>("edges", dims);
        auto frontierList = make_tensor<int>("frontier_list", dims);

        return einsum::IR::make<einsum::Definition>(
                einsum::IR::make_vec<einsum::Access>(einsum::IR::make<einsum::Access>(frontier, einsum::IR::make_vec<einsum::IndexVar>(j))),
                einsum::IR::make<einsum::ArithmeticExpression>(
                        einsum::IR::make<einsum::ReadAccess>(edges, einsum::IR::make_vec<einsum::Expression>(jExpr, kExpr)),
                        einsum::IR::make<einsum::ArithmeticExpression>(
                                einsum::IR::make<einsum::ReadAccess>(frontierList, einsum::IR::make_vec<einsum::Expression>(two, kExpr)),
                                einsum::IR::make<einsum::ComparisonExpression>(
                                        einsum::IR::make<einsum::ReadAccess>(
                                                visited,
                                                einsum::IR::make_vec<einsum::Expression>(jExpr)),
                                        zero, eq),
                                mul),
                        mul
                ),
                einsum::IR::make_vec<einsum::Reduction>(einsum::IR::make<einsum::Reduction>(k, or_, zero)));

    }

    std::shared_ptr<einsum::Definition> definition2() {
        return einsum::IR::make<einsum::Definition>(
                einsum::IR::make_vec<einsum::Access>(einsum::IR::make<einsum::Access>(
                        make_tensor<int>("round_out", {}),
                        einsum::IR::make_vec<einsum::IndexVar>()
                )),
                einsum::IR::make<einsum::ArithmeticExpression>(
                        einsum::IR::make<einsum::ReadAccess>(
                                make_tensor<int>("round_in", {}),
                                einsum::IR::make_vec<einsum::Expression>()
                        ),
                        two,
                        mul
                ),

                einsum::IR::make_vec<einsum::Reduction>()
        );
    }

    std::shared_ptr<einsum::FuncDecl> func1() {
        auto frontier = make_tensor<int>("frontier", {einsum::Type::make<einsum::ReadAccess>("N")});
        auto visited = make_tensor<int>("visited", {einsum::Type::make<einsum::ReadAccess>("N")});
        auto frontier_list = make_tensor<int>("frontier_list", {einsum::Type::make<einsum::ReadAccess>("N"),
                                                                einsum::Type::make<einsum::ReadAccess>("N")});

        return einsum::IR::make<einsum::FuncDecl>(
                "Frontier",
                einsum::IR::make_vec<einsum::TensorVar>(
                        frontier_list,
                        visited,
                        make_tensor<int>("round_in", {})
                ),
                einsum::IR::make_vec<einsum::TensorVar>(frontier, make_tensor<int>("round_out", {})),
                einsum::IR::make_vec<einsum::Definition>(definition1(), definition2())
        );
    }

protected:
    virtual void SetUp() {

    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).

    }
};

TEST_F(DumpTest, LiteralsTest) {
    auto intLit = einsum::Literal(6, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Int));
    EXPECT_EQ (intLit.dump(),  "6");

    auto boolLit = einsum::IR::make<einsum::Literal>(true, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Bool));
    EXPECT_EQ (boolLit->dump(),  "true");

    auto floatLit = einsum::IR::make<einsum::Literal>(6.0f, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Float));
    EXPECT_EQ (floatLit->dump().rfind("6.0", 0),  0);
}

TEST_F(DumpTest, ReadAccessTest1) {
    auto access = einsum::ReadAccess(A, {});
    EXPECT_EQ(access.dump(), "A");
}

TEST_F(DumpTest, ReadAccessTest2) {

    auto access = einsum::ReadAccess(A, einsum::IR::make_vec<einsum::Expression>(jExpr));
    EXPECT_EQ(access.dump(), "A[j]");
}


TEST_F(DumpTest, ReadAccessTest3) {

    auto access = einsum::ReadAccess(A, einsum::IR::make_vec<einsum::Expression>(jExpr, kExpr, jkExpr));
    EXPECT_EQ(access.dump(), "A[j][k][j + k]");
}

TEST_F(DumpTest, AccessTest) {
    EXPECT_EQ(einsum::Access(A, {}).dump(), "A");

    EXPECT_EQ(einsum::Access(A, {i}).dump(), "A[i]");

    EXPECT_EQ(einsum::Access(A, {i, j}).dump(), "A[i][j]");

    EXPECT_EQ(einsum::Access(A, {i, j, k}).dump(), "A[i][j][k]");
}

TEST_F(DumpTest, BinaryExprTest) {
    auto plus = einsum::IR::make<einsum::ArithmeticExpression>(jExpr, kExpr, add);
    EXPECT_EQ(plus->dump(), "j + k");

    auto minus = einsum::IR::make<einsum::ArithmeticExpression>(jExpr, kExpr, sub);
    EXPECT_EQ(minus->dump(), "j - k");

    auto times = einsum::IR::make<einsum::ArithmeticExpression>(plus, minus, mul);
    EXPECT_EQ(times->dump(), "(j + k) * (j - k)");

    auto half = einsum::IR::make<einsum::ArithmeticExpression>(
            times,
            two,
            //einsum::IR::make<einsum::ReadAccess>(A, einsum::IR::make_vec<einsum::Expression>()),
            div);
    EXPECT_EQ(half->dump(), "(j + k) * (j - k) / 2");

    auto minus2 = einsum::IR::make<einsum::ArithmeticExpression>(two, minus, sub);
    EXPECT_EQ(minus2->dump(), "2 - (j - k)");

    auto half2 = einsum::IR::make<einsum::ArithmeticExpression>(two, half, div);
    EXPECT_EQ(half2->dump(), "2 / ((j + k) * (j - k) / 2)");

    auto par = einsum::IR::make<einsum::ModuloExpression>(plus, two);
    EXPECT_EQ(par->dump(), "(j + k) % 2");

    auto equal = einsum::IR::make<einsum::ComparisonExpression>(jExpr, kExpr, eq);
    EXPECT_EQ(equal->dump(), "j == k");

    auto nEqual = einsum::IR::make<einsum::ComparisonExpression>(plus, kExpr, neq);
    EXPECT_EQ(nEqual->dump(), "j + k != k");

    auto more = einsum::IR::make<einsum::ComparisonExpression>(plus, minus, gt);
    EXPECT_EQ(more->dump(), "j + k > j - k");

    auto less = einsum::IR::make<einsum::ComparisonExpression>(
            einsum::IR::make<einsum::ReadAccess>(
                    A,
                    einsum::IR::make_vec<einsum::Expression>(jExpr, kExpr, jkExpr)
                    ),
            einsum::IR::make<einsum::ReadAccess>(
                    A,
                    einsum::IR::make_vec<einsum::Expression>(zero, two, jExpr)
            ), lt);
    EXPECT_EQ(less->dump(), "A[j][k][j + k] < A[0][2][j]");


    auto bool1 = einsum::IR::make<einsum::LogicalExpression>(
            more,
            yes,
            or_
            );

    EXPECT_EQ(bool1->dump(), "j + k > j - k || true");

    auto bool2 = einsum::IR::make<einsum::LogicalExpression>(
            bool1,
            no,
            and_
    );
    EXPECT_EQ(bool2->dump(), "(j + k > j - k || true) && false");

    //auto neg = einsum::IR::make<einsum::NotExpression>(bool1);
}

TEST_F(DumpTest, DefinitionTest) {
    EXPECT_EQ(definition1()->dump(),"frontier[j] = edges[j][k] * frontier_list[2][k] * (visited[j] == 0) | k:(OR, 0)");
}


TEST_F(DumpTest, FuncDeclTest1) {
    auto round_in = make_tensor<int>("round_in", {});
    auto round_out = make_tensor<int>("round_out", {});
    auto func = einsum::IR::make<einsum::FuncDecl>(
            "Round",
            einsum::IR::make_vec<einsum::TensorVar>(
                    round_in
            ),
            einsum::IR::make_vec<einsum::TensorVar>(round_out),
            einsum::IR::make_vec<einsum::Definition>(definition2())
    );
    EXPECT_EQ(func->dump(),
              "Let Round(round_in int) -> (round_out int)\n"
                   "    round_out = round_in * 2\n"
                   "End");
}

TEST_F(DumpTest, FuncDeclTest2) {
    EXPECT_EQ(func1()->dump(),
              "Let Frontier(frontier_list int[N][N], visited int[N], round_in int) -> (frontier int[N], round_out int)\n"
                   "    frontier[j] = edges[j][k] * frontier_list[2][k] * (visited[j] == 0) | k:(OR, 0)\n"
                   "    round_out = round_in * 2\n"
                   "End");
}

// Round(0)
TEST_F(DumpTest, CallTest) {
    auto args1 = einsum::IR::make_vec<einsum::Expression>(zero);
    auto call = einsum::IR::make<einsum::Call>(
                 einsum::IR::make<einsum::FuncDecl>(
                         "Round",
                         einsum::IR::make_vec<einsum::TensorVar>(
                                 make_tensor<int>("round_in", {})
                         ),
                         einsum::IR::make_vec<einsum::TensorVar>(make_tensor<int>("round_out", {})),
                         einsum::IR::make_vec<einsum::Definition>(definition2())
                 ),
                einsum::IR::make_vec<einsum::Expression>(zero)
             );
    EXPECT_EQ(call->dump(), "Round(0)");
}


// Round(0)
TEST_F(DumpTest, CallMultipleInputsTest) {
    auto round_in = make_tensor<int>("round_in", {});
    auto round_out = make_tensor<int>("round_out", {});
    auto unused_in = make_tensor<int>("unused_in", {});
    auto unused_out = make_tensor<int>("unused_out", {});
    auto args = einsum::IR::make_vec<einsum::Expression>(
            zero,
            einsum::IR::make<einsum::ReadAccess>(
                    A,
                    einsum::IR::make_vec<einsum::Expression>()));

    auto call = einsum::IR::make<einsum::Call>(
            einsum::IR::make<einsum::FuncDecl>(
                    "Round",
                    einsum::IR::make_vec<einsum::TensorVar>(
                            round_in,
                            unused_in
                    ),
                    einsum::IR::make_vec<einsum::TensorVar>(
                           round_out,
                           unused_out
                    ),
                    einsum::IR::make_vec<einsum::Definition>(definition2())
            ),
            args
    );
    EXPECT_EQ(call->dump(), "Round(0, A)");

    auto def = einsum::IR::make<einsum::Definition>(
            einsum::IR::make_vec<einsum::Access>(
                    einsum::IR::make<einsum::Access>(round_out, einsum::IR::make_vec<einsum::IndexVar>()),
                    einsum::IR::make<einsum::Access>(unused_out, einsum::IR::make_vec<einsum::IndexVar>())
            ),
            call,
            einsum::IR::make_vec<einsum::Reduction>()
    );
    EXPECT_EQ(def->dump(), "round_out, unused_out = Round(0, A)");
}

// Round*(A) | 3
TEST_F(DumpTest, CallStarRepeatTest) {
    auto round_in = make_tensor<int>("round_in", {});
    auto round_out = make_tensor<int>("round_out", {});

    auto args = einsum::IR::make_vec<einsum::Expression>(
            einsum::IR::make<einsum::ReadAccess>(
                    A,
                    einsum::IR::make_vec<einsum::Expression>()));
    auto call = einsum::IR::make<einsum::CallStarRepeat>(
                    3,
            einsum::IR::make<einsum::FuncDecl>(
                    "Round",
                    einsum::IR::make_vec<einsum::TensorVar>(
                            round_in
                    ),
                    einsum::IR::make_vec<einsum::TensorVar>(round_out),
                    einsum::IR::make_vec<einsum::Definition>(definition2())
            ),
            args
    );
    EXPECT_EQ(call->dump(), "Round*(A) | 3");

    auto def = einsum::IR::make<einsum::Definition>(
            einsum::IR::make_vec<einsum::Access>(
                    einsum::IR::make<einsum::Access>(round_out, einsum::IR::make_vec<einsum::IndexVar>())),
            call,
            einsum::IR::make_vec<einsum::Reduction>()
    );
    EXPECT_EQ(def->dump(), "round_out = Round*(A) | 3");
}


// Round*(A) | (#1 == 2)
TEST_F(DumpTest, CallStarConditionTest) {
    auto round_in = make_tensor<int>("round_in", {});
    auto round_out = make_tensor<int>("round_out", {});
    auto unused_in = make_tensor<int>("unused_in", {});
    auto unused_out = make_tensor<int>("unused_out", {});
    auto args = einsum::IR::make_vec<einsum::Expression>(
            zero,
            einsum::IR::make<einsum::ReadAccess>(
                    A,
                    einsum::IR::make_vec<einsum::Expression>()));

    auto cond = einsum::IR::make<einsum::ComparisonExpression>(
            einsum::IR::make<einsum::ReadAccess>(
                    make_tensor<int>("#1", {}),
                    einsum::IR::make_vec<einsum::Expression>()),
            two,
            eq
            );
    auto call = einsum::IR::make<einsum::CallStarCondition>(
            cond,
            einsum::IR::make<einsum::FuncDecl>(
                    "Round",
                    einsum::IR::make_vec<einsum::TensorVar>(
                            round_in,
                            unused_in
                    ),
                    einsum::IR::make_vec<einsum::TensorVar>(
                            round_out,
                            unused_out
                    ),
                    einsum::IR::make_vec<einsum::Definition>(definition2())
            ),
            args
    );
    EXPECT_EQ(call->dump(), "Round*(0, A) | (#1 == 2)");

    auto def = einsum::IR::make<einsum::Definition>(
            einsum::IR::make_vec<einsum::Access>(
                    einsum::IR::make<einsum::Access>(round_out, einsum::IR::make_vec<einsum::IndexVar>()),
                    einsum::IR::make<einsum::Access>(unused_out, einsum::IR::make_vec<einsum::IndexVar>())
            ),
            call,
            einsum::IR::make_vec<einsum::Reduction>()
    );
    EXPECT_EQ(def->dump(), "round_out, unused_out = Round*(0, A) | (#1 == 2)");
}

