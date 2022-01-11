//
// Created by Alexandra Dima on 26.12.2021.
//

#ifndef EINSUM_TACO_CODEGEN_VISITOR_H
#define EINSUM_TACO_CODEGEN_VISITOR_H

#include "einsum_taco/ir/ir.h"
namespace einsum {
    struct CodeGenVisitor : IRVisitor {

            CodeGenVisitor(std::ostream &oss, std::string module_name) : oss(oss), module_name(std::move(module_name)) {}

            void visit(const IndexVar& node) override;
            void visit(const Literal& node) override;
            void visit(const ArithmeticExpression& node) override;
            void visit(const ModuloExpression& node) override;
            void visit(const LogicalExpression& node) override;
            void visit(const ComparisonExpression& node) override;
            void visit(const NotExpression& node) override;
            void visit(const TensorVar& node) override;
            void visit(const IndexVarExpr& node) override;
            void visit(const Access& node) override;
            void visit(const ReadAccess& node) override;
            void visit(const Definition& node) override;
            void visit(const FuncDecl& node) override;
            void visit(const Call& node) override;
            void visit(const CallStarRepeat& node) override;
            void visit(const CallStarCondition& node) override;
            void visit(const Module& node) override;
    private:
        std::ostream &oss;
        std::string module_name;
        void visit_binary(const BinaryOp& node);
        void visit_unary(const UnaryOp& node);
    };

}
#endif //EINSUM_TACO_CODEGEN_VISITOR_H
