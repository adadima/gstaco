//
// Created by Alexandra Dima on 07.01.2022.
//
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/base/assert.h"
#include <iostream>

// TODO : rewrite IR to break up definitions of multiple inputs

namespace einsum {

    template<typename T>
    std::shared_ptr<T> IRRewriter::shared_from_ref(T& ref) {
        return std::dynamic_pointer_cast<T>(ref.shared_from_this());
    }

    std::shared_ptr<BinaryOp> IRRewriter::rewrite_binary(BinaryOp& node) {
        node.left = rewrite(node.left);
        node.right = rewrite(node.right);
        return shared_from_ref(node);
    }

    std::shared_ptr<UnaryOp> IRRewriter::rewrite_unary(UnaryOp& node) {
        node.expr = rewrite(node.expr);
        return shared_from_ref(node);
    }

    void IRRewriter::visit(IndexVar& node) {
        node.dimension = rewrite(node.dimension);
        index_var = shared_from_ref(node);
    }

    void  IRRewriter::visit(Literal& node) {
        expr = shared_from_ref(node);
    }

    void  IRRewriter::visit(TensorVar& node) {
        tensor = shared_from_ref(node);
    }

    void  IRRewriter::visit(IndexVarExpr& node) {
        node.indexVar = rewrite(node.indexVar);
        expr = shared_from_ref(node);
    }

    void  IRRewriter::visit(Access& node) {
        context->enter_access(shared_from_ref(node));
        node.tensor = rewrite(node.tensor);
        for(auto &indice : node.indices) {
            context->advance_access();
            indice = rewrite(indice);
        }
        access = shared_from_ref(node);
        context->exit_access();
    }

    void  IRRewriter::visit(ReadAccess& node) {
        context->enter_read_access(shared_from_ref(node));
        node.tensor = rewrite(node.tensor);
        for(auto &indice : node.indices) {
            context->advance_access();
            indice = rewrite(indice);
        }
        expr = shared_from_ref(node);
        context->exit_read_access();
    }

    void  IRRewriter::visit(Definition& node) {
        context->enter_definition(shared_from_ref(node));
        for(auto& lhs: node.lhs) {
            lhs = rewrite(lhs);
        }
        node.rhs = rewrite(node.rhs);
        for(auto& red: node.reduction_list) {
            red = rewrite(red);
        }
        def = shared_from_ref(node);
        context->exit_definition();
    }

    void  IRRewriter::visit(FuncDecl& node) {
        auto f = context->get_function(node.funcName);
        if (f) {
            func = f;
        } else {
            context->enter_function(shared_from_ref(node));
            for(auto& input: node.inputs) {
                input = rewrite(input);
            }
            for(auto& output: node.outputs) {
                output = rewrite(output);
            }
            for(auto& stmt: node.body) {
                einsum_iassert(context->func_scope() != nullptr);
                stmt = rewrite(stmt);
            }
            func = shared_from_ref(node);
            context->exit_function(func);
        }
    }

    template<typename T>
    void IRRewriter::visit_call(T& node) {
        node.function = rewrite(node.function);
        for(auto& arg: node.arguments) {
            arg = rewrite(arg);
        }
        expr = shared_from_ref(node);
    }

    void  IRRewriter::visit(Call& node) {
        visit_call(node);
    }

    void  IRRewriter::visit(CallStarRepeat& node) {
        visit_call(node);
    }

    void  IRRewriter::visit(CallStarCondition& node) {
        node.stopCondition = rewrite(node.stopCondition);
        visit_call(node);
    }

    void  IRRewriter::visit(Module& node) {
        for(auto &comp: node.decls) {
            if (comp->is_decl()) {
                comp = rewrite(shared_from_ref(comp->as_decl()));
            }
            if (comp->is_def()) {
                comp = rewrite(shared_from_ref(comp->as_def()));
            }
            if (comp->is_expr()) {
                comp = rewrite(shared_from_ref(comp->as_expr()));
            }
        }
        module = shared_from_ref(node);
    }

    void IRRewriter::visit(Reduction &node) {
        node.reductionVar = rewrite(node.reductionVar);
        node.reductionInit = rewrite(node.reductionInit);
        reduction = shared_from_ref(node);
    }

    void IRRewriter::visit(BinaryOp &node) {
        expr = rewrite_binary(node);
    }

    void IRRewriter::visit(UnaryOp &node) {
        expr = rewrite_unary(node);
    }

}