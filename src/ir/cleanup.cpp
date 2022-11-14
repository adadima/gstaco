//
// Created by Alexandra Dima on 20.01.2022.
//

#include "einsum_taco/ir/cleanup.h"
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/ir/ir.h"

namespace einsum {

    void TensorVarRewriter::visit(std::shared_ptr<TensorVar> node) {
        if (node->name == "_") {
            node_ = node;
            return;
        }
        std::shared_ptr<TensorVar> tensor;
        if (context->access_scope()) {
            tensor = context->get_write_tensor(node);
        } else if (!context->tensor_scope().empty() && context->func_scope()) {
            tensor = context->get_read_tensor(node);
        } else {
            tensor = node;
        }
        if (tensor == nullptr) {
            std::cout << "PRoblematic tensor: " << node->name << std::endl;
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

    std::vector<std::shared_ptr<Statement>> init_alloc(std::shared_ptr<TensorVar> tensor) {
        auto stmts = std::vector<std::shared_ptr<Statement>>();
        if (tensor->name != "_") {
            auto init = IR::make<Initialize>(tensor);
            stmts.push_back(init);
        }
        if (tensor->name != "_" && tensor->getOrder() > 0) {
            auto alloc = IR::make<Allocate>(tensor);
            stmts.push_back(alloc);
        }
        return stmts;
    }

    void AllocateInserter::visit(std::shared_ptr<Module> node) {
        context->enter_module(node);

        auto new_comps = std::vector<std::shared_ptr<ModuleComponent>>();

        for(auto &comp: node->decls) {
            if (comp->is_var()) {
                auto tensor = comp->as_var();

                auto init = IR::make<Initialize>(tensor);
                new_comps.push_back(init);

                if (tensor->is_global) {
                    continue;
                }

                auto alloc = IR::make<Allocate>(tensor);
                new_comps.push_back(alloc);

                continue;
            }

            if (comp->is_def()) {
                auto def = comp->as_def();
                for (auto& acc: def->lhs) {

                    auto ia = init_alloc(acc->tensor);
                    new_comps.insert(new_comps.end(), ia.begin(), ia.end());
                }
            }
            new_comps.push_back(IRRewriter::visit(comp));
        }

        node->decls = new_comps;
        node_ = node;

        context->exit_module();
    }

    void AllocateInserter::visit_decl(const std::shared_ptr<FuncDecl> &node) {
        auto new_stmts = std::vector<std::shared_ptr<Statement>>();

        for (auto &stmt: node->body) {
            if (stmt->is_def()) {
                auto def = stmt->as_def();
                for (auto& acc: def->lhs) {

                    auto ia = init_alloc(acc->tensor);
                    new_stmts.insert(new_stmts.end(), ia.begin(), ia.end());

                }
            }
            new_stmts.push_back(IRRewriter::visit(stmt));
        }

        node->body = new_stmts;
        node_ = node;
    }

    void ReductionOpGenerator::visit(std::shared_ptr<Module> node) {
        IRRewriter::visit(node);
        context->enter_module(node);
        auto new_comps = std::vector<std::shared_ptr<ModuleComponent>>();
        for (auto &op : reduction_ops) {
            new_comps.push_back(op);
        }
        new_comps.insert(new_comps.end(), node->decls.begin(), node->decls.end());
        node->decls = new_comps;
        context->exit_module();
    }

    void ReductionOpGenerator::visit(std::shared_ptr<Reduction> node) {
        IRRewriter::visit(node);
        reduction_ops.insert(node->reductionOp);
    }

    void CallRewriter::visit_decl(const std::shared_ptr<FuncDecl> &node) {
        auto new_stmts = std::vector<std::shared_ptr<Statement>>();

        for (auto &stmt: node->body) {
            if (stmt->is_def()) {
                std::cout << "OLD: " << stmt->dump() << "\n";
                inner_calls.clear();
                stmt = rewrite(stmt);

                if(inner_calls.empty()) {
                    new_stmts.push_back(stmt);
                    continue;
                }

                std::shared_ptr<Statement> new_stmt;
                for(auto&[tensor,call]: inner_calls) {
                    // A,B,C = func(...)  =>
                    // temp = func(...);
                    //
                    // temp_0 = get<0>(temp);
                    // A = temp_0;
                    //
                    // temp_1 = get<1>(temp);
                    // B = temp_1;
                    //....
                    if (tensor->is_tuple_var()) {
                        auto t = tensor->as_tuple_var();
                        new_stmt = IR::make<MultipleOutputDefinition>(t, call);
                        new_stmts.push_back(new_stmt);
                        std::cout << "REPLACE: " << new_stmt->dump() << "\n";
                        auto def = stmt->as_def();
                        for(size_t i=0; i < def->lhs.size(); i++) {
                            std::shared_ptr<TensorType> indexed_tensor_type;
                            if (auto t_ = std::dynamic_pointer_cast<Datatype>(t->type->tuple[0])) {
                                indexed_tensor_type = IR::make<TensorType>(t_, std::vector<std::shared_ptr<Expression>>());
                            } else {
                                indexed_tensor_type = std::dynamic_pointer_cast<TensorType>(t->type->tuple[0]);
                            }
//
                            auto indexed_tensor = IR::make<TensorVar>(t->name + "_" + std::to_string(i), indexed_tensor_type, false);
                            auto init = IR::make<Initialize>(indexed_tensor);
                            new_stmts.push_back(init);
                            auto index_assign = IR::make<Definition>(IR::make<Access>(indexed_tensor, std::vector<std::shared_ptr<IndexVar>>()), IR::make<TupleVarReadAccess>(t, i));
                            std::cout << "ADDITIONAL: " << index_assign->dump() << "\n";
                            new_stmts.push_back(index_assign);

                            auto assign = IR::make<Definition>(def->lhs[i], IR::make<ReadAccess>(indexed_tensor, std::vector<std::shared_ptr<Expression>>()));
                            std::cout << "ADDITIONAL: " << assign->dump() << "\n";
                            new_stmts.push_back(assign);
                        }
                    } else {
                        auto t = tensor->as_var();
                        auto init = IR::make<Initialize>(t);
                        auto alloc = IR::make<Allocate>(t);
                        auto acc = IR::make<Access>(t, std::vector<std::shared_ptr<IndexVar>>());
                        new_stmt = IR::make<Definition>(acc, call);
                        new_stmts.push_back(init);
                        new_stmts.push_back(alloc);
                        std::cout << "ADDITIONAL: " << init->dump() << "\n";
                        std::cout << "ADDITIONAL: " << alloc->dump() << "\n";
                        std::cout << "ADDITIONAL: " << new_stmt->dump() << "\n";
                        new_stmts.push_back(new_stmt);
                        std::cout << "NEW: " << stmt->dump() << "\n";
                        new_stmts.push_back(stmt);
                    }
                }
            } else {
                new_stmts.push_back(IRRewriter::visit(stmt));
            }
        }

        node->body = new_stmts;
        node_ = node;
    }

    void CallRewriter::visit_call(std::shared_ptr<Call> node) {
        std::cout << "VISIT CALL\n";
        auto out_type = std::dynamic_pointer_cast<TupleType>(node->getType());
        std::string name = "_temp_" + std::to_string(call_id);
        if (out_type->tuple.size() == 1) {
            std::shared_ptr<TensorType> type;
            if (auto t = std::dynamic_pointer_cast<Datatype>(out_type->tuple[0])) {
                type = IR::make<TensorType>(t, std::vector<std::shared_ptr<Expression>>());
            } else {
                type = std::dynamic_pointer_cast<TensorType>(out_type->tuple[0]);
            }
            auto tensor = IR::make<TensorVar>(name, type, false);
            inner_calls.insert({tensor, node});
            node_ = IR::make<ReadAccess>(tensor, std::vector<std::shared_ptr<Expression>>());
            std::cout << "Replacing call node: " << node->dump() << "\n";
            std::cout << "with: " << node_->dump() << "\n";
        } else {
//            IRRewriter::visit_call(node);
            auto output = IR::make<TupleVar>(name, out_type);
            inner_calls.insert({output, node});
            node_ = node;
//            node_ = IR::make<TupleVarReadAccess>(output, call_output_idx);
        }
        call_id += 1;

    }
}

