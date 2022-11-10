//
// Created by Alexandra Dima on 30.10.2022.
//

#ifndef EINSUM_TACO_FINCH_CODEGEN_VISITOR_H
#define EINSUM_TACO_FINCH_CODEGEN_VISITOR_H

#include "einsum_taco/ir/ir.h"
#include <functional>
#include <unordered_set>
#include <unordered_map>

namespace einsum {
    struct FinchCodeGenVisitor : DefaultIRVisitor {

        std::ostream* oss;
        std::ostream* oss_cpp;
        std::ostream* oss_h;
        std::string module_name;
        std::ostream* oss_drive;
        std::ostream* finch_compile;
        int indent_;
        int def_id = 0;
        std::unordered_map<int, std::vector<std::string>> def2tensor_args;
        std::unordered_map<int, std::string> def2func_ptr;

        FinchCodeGenVisitor(std::ostream* oss_cpp, std::ostream* oss_h, std::ostream* oss_drive, std::string module_name, bool main=true);
        ~FinchCodeGenVisitor();

//        void visit(std::shared_ptr<IndexVar> node) override;
//
        void visit(std::shared_ptr<Literal> node) override;
//
//        void visit(std::shared_ptr<TensorVar> node) override;
//
//        void visit(std::shared_ptr<IndexVarExpr> node) override;
//
//        void visit(std::shared_ptr<Access> node) override;
//
        void visit(std::shared_ptr<ReadAccess> node) override;
//
        void visit(std::shared_ptr<BinaryOp> node) override;
//
        void visit(std::shared_ptr<UnaryOp> node) override;
//
        void visit(std::shared_ptr<Definition> node) override;
//
        void visit(std::shared_ptr<Allocate> node) override;
//
//        void visit(std::shared_ptr<MemAssignment> node) override;
//
        void visit(std::shared_ptr<Initialize> node) override;
//
        void visit(std::shared_ptr<FuncDecl> node) override;
//
        void visit(std::shared_ptr<AddOperator> node) override;
//
        void visit(std::shared_ptr<MulOperator> node) override;

        void visit(std::shared_ptr<AndOperator> node) override;

        void visit(std::shared_ptr<OrOperator> node) override;

        void visit(std::shared_ptr<MinOperator> node) override;

        void visit(std::shared_ptr<ChooseOperator> node) override;
//
        void visit(std::shared_ptr<Call> node) override;

        void visit(std::shared_ptr<CallStarRepeat> node) override;

        void visit(std::shared_ptr<CallStarCondition> node) override;
//
        void visit(std::shared_ptr<Module> node) override;
//
//        void visit(std::shared_ptr<Reduction> node) override;
//
        void visit(std::shared_ptr<Datatype> node) override;
//
        void visit(std::shared_ptr<TensorType> node) override;
//
        void visit(std::shared_ptr<TupleType> node) override;
//
//        void visit(std::shared_ptr<Operator> node) override;

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

        void generate_runtime_header() const;

        void generate_runtime_source() const;

        void visit_call(const std::shared_ptr<Call>& node, const std::function<void()>& loop_generator);

        void visit_func_signature(std::shared_ptr<FuncDecl> node);
    };

    struct DefinitionVisitor : DefaultIRVisitor {
        std::ostream* oss;
        std::unordered_set<std::string> tensors;

        explicit DefinitionVisitor(std::ostream* oss) : oss(oss) {};

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<Literal> node) override;

        void visit(std::shared_ptr<Access> node) override;

        void visit(std::shared_ptr<ReadAccess> node) override;

        void visit(std::shared_ptr<BinaryOp> node) override;

        void visit(std::shared_ptr<UnaryOp> node) override;

        void visit(std::shared_ptr<Call> node) override;

        void visit(std::shared_ptr<CallStarRepeat> node) override;

        void visit(std::shared_ptr<CallStarCondition> node) override;

        void visit(std::shared_ptr<TensorVar> node) override;

    };

    struct TensorCollector : DefaultIRVisitor {
        std::unordered_set<std::string> seen;
        std::vector<std::string> tensors;

        explicit TensorCollector() {};

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<Literal> node) override;

        void visit(std::shared_ptr<ReadAccess> node) override;

        void visit(std::shared_ptr<BinaryOp> node) override;

        void visit(std::shared_ptr<UnaryOp> node) override;

        void visit_call(std::shared_ptr<Call> node);

        void visit(std::shared_ptr<Call> node) override;

        void visit(std::shared_ptr<CallStarRepeat> node) override;

        void visit(std::shared_ptr<CallStarCondition> node) override;

        void visit(std::shared_ptr<TensorVar> node) override;
    };

    struct FuncPtr2TensorArgsMapper : DefaultIRVisitor {
        std::unordered_map<int, std::vector<std::string>> def2tensor_args;
        std::unordered_map<int, std::string> def2func_ptr;
        int def_id = 0;

        explicit FuncPtr2TensorArgsMapper() {};

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<FuncDecl> node) override;

        void visit(std::shared_ptr<Module> node) override;
    };

    struct JlFunctionInitializer : DefaultIRVisitor {
        std::ostream* oss;
        int def_id = 0;

        explicit JlFunctionInitializer(std::ostream* oss) : oss(oss) {};

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<FuncDecl> node) override;

        void visit(std::shared_ptr<Module> node) override;
    };

    struct FinchCompileVisitor : DefaultIRVisitor {
        std::ostream* oss;
        std::unordered_map<int, std::vector<std::string>> def2args;
        int def_id = 0;

        FinchCompileVisitor(std::ostream* oss, std::unordered_map<int, std::vector<std::string>>& def2args) : oss(oss), def2args(def2args) {};

        void visit(std::shared_ptr<Module> node) override;

        void visit(std::shared_ptr<FuncDecl> node) override;

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<Access> node) override;

        void visit(std::shared_ptr<ReadAccess> node) override;

        void visit(std::shared_ptr<MinOperator> node) override;

        void visit(std::shared_ptr<ChooseOperator> node) override;

        void visit(std::shared_ptr<AddOperator> node) override;

        void visit(std::shared_ptr<MulOperator> node) override;

        void visit(std::shared_ptr<AndOperator> node) override;

        void visit(std::shared_ptr<OrOperator> node) override;

        void visit(std::shared_ptr<TensorVar> node) override;

        void visit(std::shared_ptr<Datatype> node) override;

        void visit(std::shared_ptr<BinaryOp> node) override;

        void visit(std::shared_ptr<UnaryOp> node) override;

        void visit(std::shared_ptr<Call> node) override;
    };

    std::string fdump(std::shared_ptr<Datatype> node);
}


#endif //EINSUM_TACO_FINCH_CODEGEN_VISITOR_H
