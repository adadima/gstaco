//
// Created by Alexandra Dima on 20.01.2022.
//

#include "einsum_taco/ir/cleanup.h"
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/ir/ir.h"

namespace einsum {

    void TensorVarRewriter::visit(std::shared_ptr<TensorVar> node) {
        std::shared_ptr<TensorVar> tensor;
        if (context->access_scope()) {
            tensor = context->get_write_tensor(node);
        } else if (!context->tensor_scope().empty() && context->func_scope()) {
            tensor = context->get_read_tensor(node);
        } else {
            tensor = node;
        }
        tensor->is_global = context->is_global(tensor);
        node_ = tensor;
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

    std::shared_ptr<Allocate> allocations_for_def(const std::shared_ptr<Definition>& def) {
        auto vars = std::vector<std::shared_ptr<TensorVar>>();
        for (auto &acc : def->lhs) {
            vars.push_back(acc->tensor);
        }
        return IR::make<Allocate>(vars);
    }

    void AllocateInserter::visit(std::shared_ptr<Module> node) {
        context->enter_module(node);
        auto new_comps = std::vector<std::shared_ptr<ModuleComponent>>();
        for(auto &comp: node->decls) {
            if (comp->is_def()) {
                auto alloc = allocations_for_def(comp->as_def());
                new_comps.push_back(alloc);
            }
            new_comps.push_back(IRRewriter::visit(comp));
        }
        node_ = node;
        context->exit_module();
    }

    void AllocateInserter::visit_decl(const std::shared_ptr<FuncDecl> &node) {
        auto new_stmts = std::vector<std::shared_ptr<Statement>>();
        for (auto &stmt: node->body) {
            if (stmt->is_def()) {
                new_stmts.push_back(allocations_for_def(stmt->as_def()));
            }
            new_stmts.push_back(IRRewriter::visit(stmt));
        }
        node->body = new_stmts;
        node_ = node;
    }
}

