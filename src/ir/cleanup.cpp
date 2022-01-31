//
// Created by Alexandra Dima on 20.01.2022.
//

#include "einsum_taco/ir/cleanup.h"
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/ir/ir.h"

namespace einsum {

    void TensorVarRewriter::visit(TensorVar &node) {
        if (context->access_scope()) {
            tensor = context->get_write_tensor(node);
        } else if (!context->tensor_scope().empty() && context->func_scope()) {
            tensor = context->get_read_tensor(node);
        } else {
            tensor = shared_from_ref(node);
        }
    }

    void TensorVarRewriter::visit(ReadAccess &node) {
        if (node.indices.empty() && context->def_scope()->has_index_var(node.tensor->name)) {
            auto ivar = IR::make<IndexVar>(node.tensor->name, nullptr);
            expr = IR::make<IndexVarExpr>(ivar);
            return;
        }
        IRRewriter::visit(node);
    }

    void FuncDeclRewriter::visit(FuncDecl &node) {
        IRRewriter::visit(node);
    }

    void IndexDimensionRewriter::visit(IndexVar &node) {
        index_var = context->get_index_var(node.getName());
        context->add_reduction_var(index_var);
    }

    void IndexDimensionRewriter::visit(IndexVarExpr &node) {
        auto ivar = context->get_index_var_expr(node.getName());
        context->add_reduction_var(ivar->indexVar);
        expr = ivar;
    }
}

