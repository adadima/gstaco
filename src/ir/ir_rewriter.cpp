//
// Created by Alexandra Dima on 07.01.2022.
//
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/base/assert.h"
#include <iostream>

// TODO : rewrite IR to break up definitions of multiple inputs

namespace einsum {

    std::shared_ptr<BinaryOp> IRRewriter::rewrite_binary(std::shared_ptr<BinaryOp> node) {
        node->left = rewrite(node->left);
        node->op = rewrite(node->op);
        node->right = rewrite(node->right);
        return node;
    }

    std::shared_ptr<UnaryOp> IRRewriter::rewrite_unary(std::shared_ptr<UnaryOp> node) {
        node->expr = rewrite(node->expr);
        node->op = rewrite(node->op);
        return node;
    }

    void IRRewriter::visit(std::shared_ptr<IndexVar> node) {
        node_ = node;
    }

    void  IRRewriter::visit(std::shared_ptr<Literal> node) {
        node_ = node;
    }

    void  IRRewriter::visit(std::shared_ptr<TensorVar> node) {
        node_ = node;
    }

    void  IRRewriter::visit(std::shared_ptr<IndexVarExpr> node) {
        node->indexVar = rewrite(node->indexVar);
        node_ = node;
    }

    void  IRRewriter::visit(std::shared_ptr<Access> node) {
        context->enter_access(node);
        node->tensor = rewrite(node->tensor);
        for(auto &indice : node->indices) {
            context->advance_access();
            indice = rewrite(indice);
        }
        node_ = node;
        context->exit_access();
    }

    void  IRRewriter::visit(std::shared_ptr<ReadAccess> node) {
        context->enter_read_access(node);
        node->tensor = rewrite(node->tensor);
        for(auto &indice : node->indices) {
            context->advance_access();
            indice = rewrite(indice);
        }
        node_ = node;
        context->exit_read_access();
    }

    void  IRRewriter::visit(std::shared_ptr<Definition> node) {
        context->enter_definition(node);
        for(auto& lhs: node->lhs) {
            lhs = rewrite(lhs);
        }
        node->rhs = rewrite(node->rhs);
        for(auto& red: node->reduction_list) {
            red = rewrite(red);
        }
        node_ = node;
        context->exit_definition(node);
    }

    void  IRRewriter::visit(std::shared_ptr<FuncDecl> node) {
        auto f = context->get_function(node->funcName);
        if (f) {
            node_ = f;
        } else {
            context->enter_function(node);
            visit_decl(node);
            context->exit_function(node);
        }
    }

    void IRRewriter::visit_call(std::shared_ptr<Call> node) {
        if (!node->function->is_builtin()) {
            node->function = rewrite(node->function);
        }

        for(auto& arg: node->arguments) {
            arg = rewrite(arg);
        }
        node_ = node;
    }

    void  IRRewriter::visit(std::shared_ptr<Call> node) {
        context->enter_call(node);
        visit_call(node);
        context->exit_call(node);
    }

    void  IRRewriter::visit(std::shared_ptr<CallStarRepeat> node) {
        context->enter_call(node);
        visit_call(node);
        context->exit_call(node);
    }

    void  IRRewriter::visit(std::shared_ptr<CallStarCondition> node) {
        context->enter_call(node);
        std::cout << "REWRITING STOP CONDITION: " << node->stopCondition->dump() << "\n";
        node->stopCondition = rewrite(node->stopCondition);
        std::cout << "NEW STOP CONDITION: " << node->stopCondition->dump() << "\n";
        node->condition_def = rewrite(node->condition_def);
        visit_call(node);
        context->exit_call(node);
    }

    void  IRRewriter::visit(std::shared_ptr<Module> node) {
        context->enter_module(node);
        for(auto &comp: node->decls) {
            if (comp == nullptr) {
                continue;
            }
            comp = rewrite(comp);
        }
        node_ = node;
        context->exit_module();
    }

    void IRRewriter::visit(std::shared_ptr<Reduction> node) {
        node->reductionVar = rewrite(node->reductionVar);
        node->reductionInit = rewrite(node->reductionInit);
        node->reductionOp = rewrite(node->reductionOp);
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<BinaryOp> node) {
        node_ = rewrite_binary(node);
    }

    void IRRewriter::visit(std::shared_ptr<UnaryOp> node) {
        node_ = rewrite_unary(node);
    }

    void IRRewriter::visit(std::shared_ptr<Datatype> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<StorageFormat> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<TensorType> node) {
        node->type = rewrite(node->type);
        for (size_t i=0; i < node->dimensions.size(); i++) {
            auto& dim = node->dimensions[i];
            dim = rewrite(dim);

            auto& format = node->formats[i];
            format = rewrite(format);
        }
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<TupleType> node) {
        for (auto &t : node->tuple) {
            t = rewrite(t);
        }
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<Operator> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<Allocate> node) {
        node->tensor = rewrite(node->tensor);
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<MemAssignment> node) {
        node->rhs = rewrite(node->rhs);
        node->lhs = rewrite(node->lhs);
        node_ = node;
    }

    std::shared_ptr<ModuleComponent> IRRewriter::visit(const std::shared_ptr<ModuleComponent>& node) {
        if (node->is_builtin()) {
            return rewrite(node->as_builtin());
        }
        if (node->is_tuple_var()) {
            return rewrite(node->as_tuple_var());
        }
        if (node->is_init()) {
            return rewrite(node->as_init());
        }
        if (node->is_decl()) {
            return rewrite(node->as_decl());
        }
        if (node->is_var()) {
            return rewrite(node->as_var());
        }
        if (node->is_def()) {
            return rewrite(node->as_def());
        }
        if (node->is_multi_def()) {
            return rewrite(node->as_multi_def());
        }
        if (node->is_expr()) {
            return rewrite(node->as_expr());
        }
        if (node->is_allocate()) {
            return rewrite(node->as_allocate());
        }
        if (node->is_mem_assign()) {
            return rewrite(node->as_mem_assign());
        }
        return node;
    }

    std::shared_ptr<Statement> IRRewriter::visit(const std::shared_ptr<Statement> &node) {

        if (node->is_allocate()) {
            return rewrite(node->as_allocate());
        }
        if (node->is_def()) {
            return rewrite(node->as_def());
        }
        if (node->is_multi_def()) {
            return rewrite(node->as_multi_def());
        }
        if (node->is_mem_assign()) {
            return rewrite(node->as_mem_assign());
        }
        return node;
    }

    void IRRewriter::visit_decl(const std::shared_ptr<FuncDecl>& node) {
        for(auto& input: node->inputs) {
            input = rewrite(input);
        }
        for(auto& output: node->outputs) {
            output = rewrite(output);
        }
        for(auto& stmt: node->body) {
            einsum_iassert(context->func_scope() != nullptr);
            stmt = rewrite(stmt);
        }
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<Initialize> node) {
        node->tensor = rewrite(node->tensor);
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<MinOperator> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<ChooseOperator> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<AddOperator> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<MulOperator> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<AndOperator> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<OrOperator> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<TupleVar> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<TupleVarReadAccess> node) {
        node->var = rewrite(node->var);
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->lhs = rewrite(node->lhs);
        node->rhs = rewrite(node->rhs);
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<BuiltinFuncDecl> node) {
        node_ = node;
    }

}