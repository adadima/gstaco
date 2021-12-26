//
// Created by Alexandra Dima on 26.12.2021.
//

#include "codegen_visitor.h"

namespace einsum {
    void CodeGenVisitor::visit(const Expression &node) {
        
    }

    void CodeGenVisitor::visit(const FuncDecl &node) {}

    void CodeGenVisitor::visit(const Definition &node) {}

    void CodeGenVisitor::visit(const Reduction &node) {}

    void CodeGenVisitor::visit(const Access &node) {}

    void CodeGenVisitor::visit(const CallStarRepeat &node) {}

    void CodeGenVisitor::visit(const CallStarCondition &node) {}

    void CodeGenVisitor::visit(const Call &node) {}

    void CodeGenVisitor::visit(const IndexVar &node) {}

    void CodeGenVisitor::visit(const TensorVar &node) {}
}