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

        CodeGenVisitor(std::ostream &oss, std::string module_name) : oss(oss), module_name(std::move(module_name)),
                                                                     indent_(0) {}

        void visit(std::shared_ptr<IndexVar> node) override;

        void visit(std::shared_ptr<Literal> node) override;

        void visit(std::shared_ptr<TensorVar> node) override;

        void visit(std::shared_ptr<IndexVarExpr> node) override;

        void visit(std::shared_ptr<Access> node) override;

        void visit(std::shared_ptr<ReadAccess> node) override;

        void visit(std::shared_ptr<BinaryOp> node) override;

        void visit(std::shared_ptr<UnaryOp> node) override;

        void visit(std::shared_ptr<Definition> node) override;
        void visit(std::shared_ptr<Allocate> node) override;

        void visit(std::shared_ptr<FuncDecl> node) override;

        void visit(std::shared_ptr<Call> node) override;

        void visit(std::shared_ptr<CallStarRepeat> node) override;

        void visit(std::shared_ptr<CallStarCondition> node) override;

        void visit(std::shared_ptr<Module> node) override;

        void visit(std::shared_ptr<Reduction> node) override;

        void visit(std::shared_ptr<Datatype> node) override;

        void visit(std::shared_ptr<TensorType> node) override;

        void visit(std::shared_ptr<TupleType> node) override;

        void visit(std::shared_ptr<Operator> node) override;

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

        void get_lambda_return(const std::shared_ptr<TupleType>& output_type, int num_outputs);

        void print_return(const std::shared_ptr<TupleType>& output_type, const std::vector<std::string>& outputs);

        void generate_tensor_template();

        void visit_call(const std::shared_ptr<Call>& node, const std::function<void()>& loop_generator);

    private:

        std::string visit_reduced_expr(const std::shared_ptr<Expression>& expr, const std::vector<std::shared_ptr<Reduction>> &reductions);

        static std::shared_ptr<Expression> reduce_expression(const std::string &init_var, std::shared_ptr<Expression> expr,
                                                      const std::shared_ptr<Operator> &op);

        void visit_tensor_declaration(const std::shared_ptr<TensorVar>& tensor);

        template<typename T>
        void visit_tensor_access(const std::shared_ptr<T>& access);

    };
}
#endif //EINSUM_TACO_CODEGEN_VISITOR_H
