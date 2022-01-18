//
// Created by Alexandra Dima on 26.12.2021.
//

#ifndef EINSUM_TACO_CODEGEN_VISITOR_H
#define EINSUM_TACO_CODEGEN_VISITOR_H

#include "einsum_taco/ir/ir.h"
namespace einsum {
    struct CodeGenVisitor : IRVisitor {

        std::ostream &oss;
        std::string module_name;
        int indent_;
        std::vector<std::string> outputs;

        CodeGenVisitor(std::ostream &oss, std::string module_name) : oss(oss), module_name(std::move(module_name)),
                                                                     indent_(0) {}

        void visit(const IndexVar &node) override;

        void visit(const Literal &node) override;

        void visit(const TensorVar &node) override;

        void visit(const IndexVarExpr &node) override;

        void visit(const Access &node) override;

        void visit(const ReadAccess &node) override;

        void visit(const BinaryOp &node) override;

        void visit(const UnaryOp &node) override;

        void visit(const Definition &node) override;

        void visit(const FuncDecl &node) override;

        void visit(const Call &node) override;

        void visit(const CallStarRepeat &node) override;

        void visit(const CallStarCondition &node) override;

        void visit(const Module &node) override;

        void visit(const Reduction &node) override;

        int &get_indent_lvl() {
            return indent_;
        }

        void indent() {
            get_indent_lvl()++;
        }

        void unindent() {
            get_indent_lvl()--;
        }

        std::string get_indent() {
            return std::string(get_indent_lvl() * 4, ' ');
        }

        void generate_for_loop(const std::string& var, const std::shared_ptr<Expression>& dim);

        void generate_while_loop(const std::shared_ptr<Expression>& condition);

        void get_lambda_return(std::string output_type, int num_outputs);

    private:

        std::string visit_reduced_expr(const std::shared_ptr<Expression>& expr, const std::vector<std::shared_ptr<Reduction>> &reductions);

        static std::shared_ptr<Expression> reduce_expression(const std::string &init_var, std::shared_ptr<Expression> expr,
                                                      const std::shared_ptr<Operator> &op);

    };
}
#endif //EINSUM_TACO_CODEGEN_VISITOR_H
