//
// Created by Alexandra Dima on 26.12.2021.
//

#ifndef EINSUM_TACO_CODEGEN_VISITOR_H
#define EINSUM_TACO_CODEGEN_VISITOR_H

#include "einsum_taco/ir/ir.h"
namespace einsum {
    struct CodeGenVisitor : IRVisitor {
            void visit(const Expression &node) override;

            void visit(const FuncDecl &node) override;

            void visit(const Definition &node) override;

            void visit(const Reduction &node) override;

            void visit(const Access &node) override;

            void visit(const CallStarRepeat &node) override;

            void visit(const CallStarCondition &node) override;

            void visit(const Call &node) override;

            void visit(const IndexVar &node) override;

            void visit(const TensorVar &node) override;
    };

}
#endif //EINSUM_TACO_CODEGEN_VISITOR_H
