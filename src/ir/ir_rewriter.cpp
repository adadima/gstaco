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
        node->dimension = rewrite(node->dimension);
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
        context->exit_definition();
    }

    void  IRRewriter::visit(std::shared_ptr<FuncDecl> node) {
        auto f = context->get_function(node->funcName);
        if (f) {
            node_ = f;
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
            node_ = node;
            context->exit_function(node);
        }
    }

    template<typename T>
    void IRRewriter::visit_call(T& node) {
        node->function = rewrite(node->function);
        for(auto& arg: node->arguments) {
            arg = rewrite(arg);
        }
        node_ = node;
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
        context->enter_module(node);
        for(auto &comp: node->decls) {
            if (comp->is_decl()) {
                comp = rewrite(comp->as_decl());
            }
            if (comp->is_var()) {
                comp = rewrite(comp->as_var());
            }
            if (comp->is_def()) {
                comp = rewrite(comp->as_def());
            }
            if (comp->is_expr()) {
                comp = rewrite(comp->as_expr());
            }
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

    void IRRewriter::visit(std::shared_ptr<TensorType> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<TupleType> node) {
        node_ = node;
    }

    void IRRewriter::visit(std::shared_ptr<Operator> node) {
        node_ = node;
    }

}