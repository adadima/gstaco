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
        int indent_;
        int def_id = 0;
        std::unordered_map<int, std::vector<std::shared_ptr<TensorVar>>> def2tensor_args;
        std::unordered_map<int, std::string> def2func_ptr;
        std::unordered_map<int, bool> def2needs_finch;

        FinchCodeGenVisitor(std::ostream* oss_cpp, std::ostream* oss_h, std::ostream* oss_drive, std::string module_name, bool main=true);
        ~FinchCodeGenVisitor();

        std::string name() override {
            return "FinchCodeGenVisitor";
        }
//        void visit(std::shared_ptr<IndexVar> node) override;
//
        void visit(std::shared_ptr<Literal> node) override;
//
        void visit(std::shared_ptr<TensorVar> node) override;
//
//        void visit(std::shared_ptr<IndexVarExpr> node) override;
//
        void visit(std::shared_ptr<Access> node) override;
//
        void visit(std::shared_ptr<ReadAccess> node) override;

        void visit(std::shared_ptr<TupleVarReadAccess> node) override;
//
        void visit(std::shared_ptr<BinaryOp> node) override;
//
        void visit(std::shared_ptr<UnaryOp> node) override;
//
        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<MultipleOutputDefinition> node) override;
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

        void generate_while_loop(const std::shared_ptr<CallStarCondition>& node);

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
        std::unordered_map<int, bool> def2needs_finch;

        explicit DefinitionVisitor(std::ostream* oss, std::unordered_map<int, bool>& def2needs_finch) : oss(oss), def2needs_finch(def2needs_finch) {};

        std::string name() override {
            return "DefinitionVisitor";
        }

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<MultipleOutputDefinition> node) override;

        void visit(std::shared_ptr<Literal> node) override;

        void visit(std::shared_ptr<IndexVarExpr> node) override;

        void visit(std::shared_ptr<Access> node) override;

        void visit(std::shared_ptr<ReadAccess> node) override;

        void visit(std::shared_ptr<TupleVarReadAccess> node) override;

        void visit(std::shared_ptr<BinaryOp> node) override;

        void visit(std::shared_ptr<UnaryOp> node) override;

        void visit(std::shared_ptr<Call> node) override;

        void visit(std::shared_ptr<CallStarRepeat> node) override;

        void visit(std::shared_ptr<CallStarCondition> node) override;

        void visit(std::shared_ptr<TensorVar> node) override;

    };

    struct FinchDefinitionChecker : DefaultIRVisitor {
        bool needs_finch = false;

        FinchDefinitionChecker() = default;

        std::string name() override {
            return "FinchDefinitionChecker";
        }

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<MultipleOutputDefinition> node) override;

        void visit(std::shared_ptr<Literal> node) override;

        void visit(std::shared_ptr<IndexVarExpr> node) override;

        void visit(std::shared_ptr<Access> node) override;

        void visit(std::shared_ptr<ReadAccess> node) override;

        void visit(std::shared_ptr<TupleVarReadAccess> node) override;

        void visit(std::shared_ptr<CallStarCondition> node) override;

        void visit(std::shared_ptr<Call> node) override;

        void visit(std::shared_ptr<CallStarRepeat> node) override;

        void visit_call(std::shared_ptr<Call> node);

        void visit(std::shared_ptr<TensorVar> node) override;

        void visit(std::shared_ptr<BinaryOp> node) override;

        void visit(std::shared_ptr<UnaryOp> node) override;
    };

    struct NeedsFinchVisitor : DefaultIRVisitorUnsafe {
        std::unordered_map<int, bool> def2needs_finch;
        int def_id = 0;

        NeedsFinchVisitor() = default;

        std::string name() override {
            return "NeedsFinchVisitor";
        }

        void visit(std::shared_ptr<Definition> node) override;
    };

    struct TensorCollector : DefaultIRVisitor {
        std::unordered_set<std::string> seen;
        std::vector<std::shared_ptr<TensorVar>> tensors;

        explicit TensorCollector() {};

        std::string name() override {
            return "TensorCollector";
        }

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<MultipleOutputDefinition> node) override;

        void visit(std::shared_ptr<Literal> node) override;

        void visit(std::shared_ptr<IndexVarExpr> node) override;

        void visit(std::shared_ptr<ReadAccess> node) override;

        void visit(std::shared_ptr<TupleVarReadAccess> node) override;

        void visit(std::shared_ptr<BinaryOp> node) override;

        void visit(std::shared_ptr<UnaryOp> node) override;

        void visit_call(std::shared_ptr<Call> node);

        void visit(std::shared_ptr<Call> node) override;

        void visit(std::shared_ptr<CallStarRepeat> node) override;

        void visit(std::shared_ptr<CallStarCondition> node) override;

        void visit(std::shared_ptr<TensorVar> node) override;
    };

    struct FuncPtr2TensorArgsMapper : DefaultIRVisitorUnsafe {
        std::unordered_map<int, std::vector<std::shared_ptr<TensorVar>>> def2tensor_args;
        std::unordered_map<int, std::string> def2func_ptr;
        int def_id = 0;

        explicit FuncPtr2TensorArgsMapper() {};

        std::string name() override {
            return "FuncPtr2TensorArgsMapper";
        }

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<MultipleOutputDefinition> node) override;

        void visit(std::shared_ptr<FuncDecl> node) override;

        void visit(std::shared_ptr<Module> node) override;

        void visit(std::shared_ptr<CallStarCondition> node) override;
    };

    struct JlFunctionInitializer : DefaultIRVisitorUnsafe {
        std::ostream* oss;
        int def_id = 0;

        explicit JlFunctionInitializer(std::ostream* oss) : oss(oss) {};

        std::string name() override {
            return "JlFunctionInitializer";
        }

        void visit(std::shared_ptr<Definition> node) override;

        void visit(std::shared_ptr<MultipleOutputDefinition> node) override;

        void visit(std::shared_ptr<FuncDecl> node) override;

        void visit(std::shared_ptr<Module> node) override;
    };

    struct FinchCompileVisitor : DefaultIRVisitorUnsafe {
        std::ostream* oss;
        std::ostream* junk;
        std::ostream* finch;
        std::unordered_map<int, std::vector<std::shared_ptr<TensorVar>>> def2args;
        int def_id = 0;
        std::string module;
        std::unordered_map<int, bool> def2needs_finch;
        bool explore_mode = false;

        FinchCompileVisitor(std::string& module, std::ostream* finch_oss, std::ostream* junk_oss, std::unordered_map<int, std::vector<std::shared_ptr<TensorVar>>>& def2args, std::unordered_map<int, bool>& def2needs_finch) : oss(finch_oss), finch(finch_oss), junk(junk_oss), def2args(def2args), module(module), def2needs_finch(def2needs_finch) {};

        std::string name() override {
            return "FinchCompileVisitor";
        }

        void visit(std::shared_ptr<Module> node) override;

        void visit(std::shared_ptr<FuncDecl> node) override;

        void visit(std::shared_ptr<Definition> node) override;
        void visit(std::shared_ptr<MultipleOutputDefinition> node) override;

        void visit(std::shared_ptr<Literal> node) override;

        void visit(std::shared_ptr<IndexVarExpr> node) override;

        void visit(std::shared_ptr<Access> node) override;

        void visit(std::shared_ptr<ReadAccess> node) override;

        void visit(std::shared_ptr<BuiltinFuncDecl> node) override;

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

        void visit(std::shared_ptr<CallStarCondition> node) override;

        void visit(std::shared_ptr<CallStarRepeat> node) override;
    };

    std::string fdump(std::shared_ptr<Datatype> node);
}


#endif //EINSUM_TACO_FINCH_CODEGEN_VISITOR_H
