//
// Created by Alexandra Dima on 26.12.2021.
//

#ifndef EINSUM_TACO_CODEGEN_VISITOR_H
#define EINSUM_TACO_CODEGEN_VISITOR_H

#include "einsum_taco/ir/ir.h"
namespace einsum {
    struct CodeGenVisitor : IRVisitor {

        std::ostream* oss;
        std::ostream* oss_cpp;
        std::ostream* oss_h;
        std::string module_name;
        std::ostream* oss_drive;
        int indent_;

        CodeGenVisitor(std::ostream* oss_cpp, std::ostream* oss_h, std::ostream* oss_drive, std::string module_name, bool main=true) : oss(oss_cpp), oss_cpp(oss_cpp), oss_h(oss_h), module_name(std::move(module_name)),
                                                                     indent_(0), oss_drive(oss_drive) {}

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

        void visit(std::shared_ptr<MemAssignment> node) override;

        void visit(std::shared_ptr<Initialize> node) override;

        void visit(std::shared_ptr<FuncDecl> node) override;

        void visit(std::shared_ptr<AddOperator> node) override;

        void visit(std::shared_ptr<MulOperator> node) override;

        void visit(std::shared_ptr<AndOperator> node) override;

        void visit(std::shared_ptr<OrOperator> node) override;

        void visit(std::shared_ptr<MinOperator> node) override;

        void visit(std::shared_ptr<ChooseOperator> node) override;

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

        void generate_tensor_template() const;

        void generate_driver_code() const;


        void visit_call(const std::shared_ptr<Call>& node, const std::function<void()>& loop_generator);

        void visit_func_signature(std::shared_ptr<FuncDecl> node);
    private:

        std::string visit_reduced_expr(const std::shared_ptr<Expression>& expr, const std::vector<std::shared_ptr<Reduction>> &reductions);

        static std::shared_ptr<Expression> reduce_expression(const std::string &init_var, std::shared_ptr<Expression> expr,
                                                      const std::shared_ptr<BuiltinFuncDecl> &op);

        template<typename T>
        void visit_tensor_access(const std::shared_ptr<T>& access);

    };
}
#endif //EINSUM_TACO_CODEGEN_VISITOR_H
