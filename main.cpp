#include <iostream>
#include "include/einsum_taco/ir/ir.h"

int main(int argc, char *argv[]) {
    using namespace einsum;

    auto intLit = Literal(6, std::make_shared<Datatype>(Datatype::Kind::Int));
    std::cout << "Int Literal: " << intLit.dump() << "\n";
    auto boolLit = Literal(true, std::make_shared<Datatype>(Datatype::Kind::Bool));
    std::cout << "Bool Literal: " << boolLit.dump() << "\n";
    auto floatLit = Literal(6.0f, std::make_shared<Datatype>(Datatype::Kind::Float));
    std::cout << "Float Literal: " << floatLit.dump() << "\n";

    std::vector<std::shared_ptr<DimensionType>> empty = {};
    auto tType = std::make_shared<TensorType>(std::make_shared<Datatype>(Datatype::Kind::Int), empty);
    auto tvar = std::make_shared<TensorVar>("A", tType);
    auto access = ReadAccess(tvar, {});
    std::cout << "Access tensor: " << access.dump() << "\n";

    std::vector<std::shared_ptr<DimensionType>> oneD = {std::make_shared<FixedDimension>(2)};
    tType = std::make_shared<TensorType>(std::make_shared<Datatype>(Datatype::Kind::Int), oneD);
    tvar = std::make_shared<TensorVar>("A", tType);
    auto ind = std::make_shared<Literal>(0, std::make_shared<Datatype>(Datatype::Kind::Int));
    auto var = std::make_shared<IndexVar>("j", 10);
    access = ReadAccess(tvar, {ind});
    auto writeAccess = Access(tvar, {var});
    std::cout << "Access tensor: " << access.dump() << "\n";
    std::cout << "Write access tensor: " << writeAccess.dump() << "\n";

    auto indVar1 = IR::make<IndexVarExpr>(IR::make<IndexVar>("j", 0));
    auto indVar2 = IR::make<IndexVarExpr>(IR::make<IndexVar>("k", 0));
    auto indExpr = std::make_shared<ArithmeticExpression>(indVar1, indVar2, std::make_shared<AddOp>());

    std::vector<std::shared_ptr<DimensionType>> threeD = {std::make_shared<FixedDimension>(2), std::make_shared<FixedDimension>(3), std::make_shared<FixedDimension>(1)};
    tType = std::make_shared<TensorType>(std::make_shared<Datatype>(Datatype::Kind::Float), threeD);
    tvar = std::make_shared<TensorVar>("A", tType);
    auto ind1 = std::make_shared<Literal>(0, std::make_shared<Datatype>(Datatype::Kind::Int));
    auto ind2 = IR::make<IndexVarExpr>(IR::make<IndexVar>("i", 3));
    access = ReadAccess(tvar, {ind1, ind2, indExpr});
    std::cout << "Access tensor: " << access.dump() << "\n";

    // edges -> int[10][5]
    // frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(OR, 0)
    auto zero = IR::make<Literal>(0, Type::make<Datatype>(Datatype::Kind::Int));
    auto type = Type::make<TensorType>(
                            Type::make<Datatype>(
                                    Datatype::Kind::Int
                                    ),
                            Type::make_vec<DimensionType>(
                                    Type::make<FixedDimension>(10)
                                            )
                            );

    // define indices
    auto j = IR::make<IndexVar>("j", 10);
    auto k = IR::make<IndexVarExpr>(IR::make<IndexVar>("k", 5));
    auto round = IR::make<IndexVarExpr>(IR::make<IndexVar>("round", 15));
    auto jr = IR::make<IndexVarExpr>(IR::make<IndexVar>("j", 10));

    // define tensor vars
    auto frontier = IR::make<TensorVar>("frontier", type);
    auto visited = IR::make<TensorVar>("visited", type);
    auto edges = IR::make<TensorVar>(
            "edges",
            Type::make<TensorType>(
                    Type::make<Datatype>(
                            Datatype::Kind::Int
                    ),
                    Type::make_vec<DimensionType>(
                            Type::make<FixedDimension>(10),
                            Type::make<FixedDimension>(5)
                    )
            ));
    auto frontierList = IR::make<TensorVar>(
            "frontier_list",
            Type::make<TensorType>(
                    Type::make<Datatype>(
                            Datatype::Kind::Int
                    ),
                    Type::make_vec<DimensionType>(
                            Type::make<FixedDimension>(15),
                            Type::make<FixedDimension>(5)
                    )
            ));

    // define operators
    auto mul = std::make_shared<MulOp>();
    auto or_ = std::make_shared<OrOp>();
    auto eq = std::make_shared<EqOp>();

    std::vector<std::shared_ptr<Reduction>> reductions = {IR::make<Reduction>(k->indexVar, or_, zero)};

    auto def = IR::make<Definition>(
            IR::make<Access>(frontier, IR::make_vec<IndexVar>(j)),
            IR::make<ArithmeticExpression>(
                    IR::make<ReadAccess>(edges, IR::make_vec<Expression>(jr, k)),
                    IR::make<ArithmeticExpression>(
                            IR::make<ReadAccess>(frontierList, IR::make_vec<Expression>(round, k)),
                            IR::make<ComparisonExpression>(
                                    IR::make<ReadAccess>(
                                            visited,
                                            IR::make_vec<Expression>(jr)),
                                            zero, eq),
                            mul),
                    mul
                    ),
            reductions);
    std::cout << "frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(OR, 0)\n" << def->dump() << "\n";

}