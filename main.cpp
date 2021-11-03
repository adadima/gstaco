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

    auto indVar1 = std::make_shared<IndexVarExpr>("j", 0);
    auto indVar2 = std::make_shared<IndexVarExpr>("k", 0);
    auto indExpr = std::make_shared<ArithmeticExpression>(indVar1, indVar2, std::make_shared<AddOp>());

    std::vector<std::shared_ptr<DimensionType>> threeD = {std::make_shared<FixedDimension>(2), std::make_shared<FixedDimension>(3), std::make_shared<FixedDimension>(1)};
    tType = std::make_shared<TensorType>(std::make_shared<Datatype>(Datatype::Kind::Float), threeD);
    tvar = std::make_shared<TensorVar>("A", tType);
    auto ind1 = std::make_shared<Literal>(0, std::make_shared<Datatype>(Datatype::Kind::Int));
    auto ind2 = std::make_shared<IndexVarExpr>("i", 3);
    access = ReadAccess(tvar, {ind1, ind2, indExpr});
    std::cout << "Access tensor: " << access.dump() << "\n";

    // edges -> int[10][5]
    // frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(OR, 0)
    auto zero = std::make_shared<Literal>(0, std::make_shared<Datatype>(Datatype::Kind::Int));
    std::vector<std::shared_ptr<DimensionType>> edgesDims = {std::make_shared<FixedDimension>(10),
                                                             std::make_shared<FixedDimension>(5)};
    std::vector<std::shared_ptr<DimensionType>> frontierListDims = {std::make_shared<FixedDimension>(15),
                                                                    std::make_shared<FixedDimension>(5)};
    std::vector<std::shared_ptr<DimensionType>> dims = {std::make_shared<FixedDimension>(10)};

    auto type = std::make_shared<TensorType>(std::make_shared<Datatype>(Datatype::Kind::Int), dims);
    auto edgesType = std::make_shared<TensorType>(std::make_shared<Datatype>(Datatype::Kind::Int), edgesDims);
    auto frontierListType = std::make_shared<TensorType>(std::make_shared<Datatype>(Datatype::Kind::Int), frontierListDims);

    // define indices
    auto j = std::make_shared<IndexVar>("j", 10);
    auto k = std::make_shared<IndexVarExpr>("k", 5);
    auto round = std::make_shared<IndexVarExpr>("round", 15);
    auto jr = std::make_shared<IndexVarExpr>("j", 10);

    // define tensor vars
    auto frontier = std::make_shared<TensorVar>("frontier", type);
    auto visited = std::make_shared<TensorVar>("visited", type);
    auto edges = std::make_shared<TensorVar>("edges", edgesType);
    auto frontierList = std::make_shared<TensorVar>("frontier_list", frontierListType);

    std::vector<std::shared_ptr<Expression>> acc1 = {jr, k};
    std::vector<std::shared_ptr<Expression>> acc2 = {round, k};
    std::vector<std::shared_ptr<Expression>> acc3 = {jr};
    auto edgesAcc = std::make_shared<ReadAccess>(edges, acc1);
    auto flAcc = std::make_shared<ReadAccess>(frontierList, acc2);
    auto visitedAcc = std::make_shared<ReadAccess>(visited, acc3);

    // define lhs
    std::vector<std::shared_ptr<IndexVar>> leftIndices = {j};
    auto lhs = std::make_shared<Access>(frontier, leftIndices);

    std::vector<std::shared_ptr<IndexVar>> rightIndices = {k, round};
    auto mul = std::make_shared<MulOp>();
    auto or_ = std::make_shared<OrOp>();
    auto eq = std::make_shared<EqOp>();

    auto logicExpr = std::make_shared<ComparisonExpression>(visitedAcc, zero, eq);
    auto expr2 = std::make_shared<ArithmeticExpression>(flAcc, logicExpr, mul);
    auto rhs = std::make_shared<ArithmeticExpression>(edgesAcc, expr2, mul);

    auto reduction = std::make_shared<Reduction>(k, or_, zero);
    std::cout << or_->reductionSign << "\n";
    std::map<std::shared_ptr<IndexVar>, std::shared_ptr<Reduction>> reductions;
    reductions[k] = reduction;
    auto def = std::make_shared<Definition>(lhs, leftIndices, rhs, rightIndices, reductions);
    std::cout << "frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(OR, 0)\n" << def->dump() << "\n";

}