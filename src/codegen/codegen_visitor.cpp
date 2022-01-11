//
// Created by Alexandra Dima on 26.12.2021.
//

#include "einsum_taco/codegen/codegen_visitor.h"

namespace einsum {
    void CodeGenVisitor::visit(const IndexVar& node) {
        oss << node.dump();
    }
    void CodeGenVisitor::visit(const Literal& node) {
        oss << node.dump();
    }
    void CodeGenVisitor::visit(const ArithmeticExpression& node) {
        visit_binary(node);
    }
    void CodeGenVisitor::visit(const ModuloExpression& node) {
        visit_binary(node);
    }
    void CodeGenVisitor::visit(const LogicalExpression& node) {
        visit_binary(node);
    }
    void CodeGenVisitor::visit(const ComparisonExpression& node) {
        visit_binary(node);
    }
    void CodeGenVisitor::visit(const NotExpression& node) {
        visit_unary(node);
    }
    void CodeGenVisitor::visit(const TensorVar& node) {
        oss << node.dump();
    }
    void CodeGenVisitor::visit(const IndexVarExpr& node) {
        oss << node.dump();
    }
    void CodeGenVisitor::visit(const Access& node) {
        oss << node.dump();
    }
    void CodeGenVisitor::visit(const ReadAccess& node) {
        oss << node.dump();
    }

    // frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(OR, 0)
    void CodeGenVisitor::visit(const Definition& node) {

    }
    void CodeGenVisitor::visit(const FuncDecl& node) {}
    void CodeGenVisitor::visit(const Call& node) {}
    void CodeGenVisitor::visit(const CallStarRepeat& node) {}
    void CodeGenVisitor::visit(const CallStarCondition& node) {}
    void CodeGenVisitor::visit(const Module& node) {}

    void CodeGenVisitor::visit_binary(const BinaryOp &node) {
        oss << node.dump();
    }

    void CodeGenVisitor::visit_unary(const UnaryOp &node) {
        oss << node.dump();
    }
}