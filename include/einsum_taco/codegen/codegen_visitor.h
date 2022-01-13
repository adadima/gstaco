//
// Created by Alexandra Dima on 26.12.2021.
//

#ifndef EINSUM_TACO_CODEGEN_VISITOR_H
#define EINSUM_TACO_CODEGEN_VISITOR_H

#include "einsum_taco/ir/ir.h"
namespace einsum {
    struct CodeGenVisitor : IRVisitor {

            //std::vector<std::shared_ptr<Reduction>> reductions;
            int indent_;

            CodeGenVisitor(std::ostream &oss, std::string module_name) : oss(oss), module_name(std::move(module_name)), indent_(0) {}

            void visit(const IndexVar& node) override;
            void visit(const Literal& node) override;
            void visit(const TensorVar& node) override;
            void visit(const IndexVarExpr& node) override;
            void visit(const Access& node) override;
            void visit(const ReadAccess& node) override;
            void visit(const BinaryOp& node) override;
            void visit(const UnaryOp& node) override;
            void visit(const Definition& node) override;
            void visit(const FuncDecl& node) override;
            void visit(const Call& node) override;
            void visit(const CallStarRepeat& node) override;
            void visit(const CallStarCondition& node) override;
            void visit(const Module& node) override;
            void visit(const Reduction& node) override;

            int& get_indent_lvl() {
                return indent_;
            }

            void indent() {
                get_indent_lvl() ++;
            }

            void unindent() {
                get_indent_lvl() --;
            }

            std::string get_indent() {
                return std::string(get_indent_lvl(), '\t');
            }

    private:
        std::ostream &oss;
        std::string module_name;
        void visit_reduced_expr(const Expression& expr, const std::vector<std::shared_ptr<Reduction>>& reductions);
    };

}
#endif //EINSUM_TACO_CODEGEN_VISITOR_H
