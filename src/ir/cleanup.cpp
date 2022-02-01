//
// Created by Alexandra Dima on 20.01.2022.
//

#include "einsum_taco/ir/cleanup.h"
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/ir/ir.h"

namespace einsum {

    void TensorVarRewriter::visit(std::shared_ptr<TensorVar> node) {
        if (context->access_scope()) {
            node_ = context->get_write_tensor(node);
        } else if (!context->tensor_scope().empty() && context->func_scope()) {
            node_ = context->get_read_tensor(node);
        } else {
            node_ = node;
        }
    }

    void TensorVarRewriter::visit(std::shared_ptr<ReadAccess> node) {
        if (node->indices.empty() && context->def_scope()->has_index_var(node->tensor->name)) {
            auto ivar = IR::make<IndexVar>(node->tensor->name, nullptr);
            node_ = IR::make<IndexVarExpr>(ivar);
            return;
        }
        IRRewriter::visit(node);
    }

    void FuncDeclRewriter::visit(std::shared_ptr<FuncDecl> node) {
        IRRewriter::visit(node);
    }

    void IndexDimensionRewriter::visit(std::shared_ptr<IndexVar> node) {
        auto ivar = context->get_index_var(node->getName());
        node_ = ivar;
        context->add_reduction_var(ivar);
    }

    void IndexDimensionRewriter::visit(std::shared_ptr<IndexVarExpr> node) {
        auto ivar = context->get_index_var_expr(node->getName());
        context->add_reduction_var(ivar->indexVar);
        node_ = ivar;
    }
}

