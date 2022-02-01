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
        node->right = rewrite(node->right);
        return node;
    }

    std::shared_ptr<UnaryOp> IRRewriter::rewrite_unary(std::shared_ptr<UnaryOp> node) {
        node->expr = rewrite(node->expr);
        return node;
    }

    void IRRewriter::visit(std::shared_ptr<IndexVar> node) {
        node->dimension = rewrite(node->dimension);
        index_var = node;
    }

    void  IRRewriter::visit(std::shared_ptr<Literal> node) {
        expr = node;
    }

    void  IRRewriter::visit(std::shared_ptr<TensorVar> node) {
        tensor = node;
    }

    void  IRRewriter::visit(std::shared_ptr<IndexVarExpr> node) {
        node->indexVar = rewrite(node->indexVar);
        expr = node;
    }

    void  IRRewriter::visit(std::shared_ptr<Access> node) {
        context->enter_access(node);
        node->tensor = rewrite(node->tensor);
        for(auto &indice : node->indices) {
            context->advance_access();
            indice = rewrite(indice);
        }
        access = node;
        context->exit_access();
    }

    void  IRRewriter::visit(std::shared_ptr<ReadAccess> node) {
        context->enter_read_access(node);
        node->tensor = rewrite(node->tensor);
        for(auto &indice : node->indices) {
            context->advance_access();
            indice = rewrite(indice);
        }
        expr = node;
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
        def = node;
        context->exit_definition();
    }

    void  IRRewriter::visit(std::shared_ptr<FuncDecl> node) {
        auto f = context->get_function(node->funcName);
        if (f) {
            func = f;
        } else {
            context->enter_function(node);
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
            func = node;
            context->exit_function(func);
        }
    }

    template<typename T>
    void IRRewriter::visit_call(T& node) {
        node->function = rewrite(node->function);
        for(auto& arg: node->arguments) {
            arg = rewrite(arg);
        }
        expr = node;
    }

    void  IRRewriter::visit(std::shared_ptr<Call> node) {
        visit_call(node);
    }

    void  IRRewriter::visit(std::shared_ptr<CallStarRepeat> node) {
        visit_call(node);
    }

    void  IRRewriter::visit(std::shared_ptr<CallStarCondition> node) {
        node->stopCondition = rewrite(node->stopCondition);
        visit_call(node);
    }

    void  IRRewriter::visit(std::shared_ptr<Module> node) {
        for(auto &comp: node->decls) {
            if (comp->is_decl()) {
                comp = rewrite(comp->as_decl());
            }
            if (comp->is_def()) {
                comp = rewrite(comp->as_def());
            }
            if (comp->is_expr()) {
                comp = rewrite(comp->as_expr());
            }
        }
        module = node;
    }

    void IRRewriter::visit(std::shared_ptr<Reduction> node) {
        node->reductionVar = rewrite(node->reductionVar);
        node->reductionInit = rewrite(node->reductionInit);
        reduction = node;
    }

    void IRRewriter::visit(std::shared_ptr<BinaryOp> node) {
        expr = rewrite_binary(node);
    }

    void IRRewriter::visit(std::shared_ptr<UnaryOp> node) {
        expr = rewrite_unary(node);
    }

}