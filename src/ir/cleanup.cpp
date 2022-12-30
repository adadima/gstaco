//
// Created by Alexandra Dima on 20.01.2022.
//

#include "einsum_taco/ir/cleanup.h"
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/ir/ir.h"

namespace einsum {

    int parse_index(const std::string& var) {
        auto nth = std::stoi(var.substr(1));
        return nth - 1;
    }

    std::string parse_variable_name(int nth) {
        return "out" + std::to_string(nth);
    }

    void TensorVarRewriter::visit(std::shared_ptr<TensorVar> node) {
        if (node->name == "_") {
            node_ = node;
            return;
        }
        std::shared_ptr<TensorVar> tensor;
        if (context->read_access_scope()) {
            tensor = context->get_read_tensor(node);
        } else if (context->access_scope()) {
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
        if (tensor->name.rfind('#', 0) == 0) {
//            std::cout << "Tensor: " << tensor->name << "\n";
//            std::cout << "TUPLE TYPE: " << context->call_scope()->function->getOutputType()->dump() << "\n";
            int idx = parse_index(tensor->name);
//            std::cout << "Idx: " << idx << "\n";
            tensor->name = parse_variable_name(idx);
//            std::cout << "New name: " << tensor->name << "\n";
//            std:: cout << "Call: " << context->call_scope()->dump() << "\n";
//            std::cout << "Function: " << context->call_scope()->function->dump() << "\n";
//            std::cout << "Associated input size: " << context->call_scope()->function->inputs.size() << "\n";
            tensor->type = context->call_scope()->function->inputs[idx]->type;
//            std::cout << "Element " << idx << " of tuple type: " << tensor->type->dump() << "\n";
        }
        node_ = tensor;
    }

    void TensorVarRewriter::visit(std::shared_ptr<ReadAccess> node) {
        if (node->indices.empty() && (in_access || context->def_scope()->has_index_var(node->tensor->name)) ) {
            auto ivar = IR::make<IndexVar>(node->tensor->name);
            context->def_scope()->indexVars.insert(node->tensor->name);
            node_ = IR::make<IndexVarExpr>(ivar);
            std::cout << "REPLACING READ ACCESS " << node->dump() << "WITH INDEX VAR: " << node_->dump() << "\n";
            return;
        }
        IRRewriter::visit(node);
    }

    void TensorVarRewriter::visit(std::shared_ptr<Access> node) {
        in_access = true;
        IRRewriter::visit(node);
        in_access = false;
    }

    void AccessRewriter::visit(std::shared_ptr<Access> node) {
        index_vars.clear();
        for (auto& idx: node->indices) {
            idx->accept(this);
        }
        node->index_vars.insert(node->index_vars.end(), index_vars.begin(), index_vars.end());
        node_ = node;
    }

    void AccessRewriter::visit(std::shared_ptr<IndexVarExpr> node) {
        index_vars.push_back(node);
        std::cout << "ADDING INDEX VAR EXPR TO SET: " << node->dump() << "\n";
        node_ = node;
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
        for (auto &[sign,op] : reduction_ops) {
            std::cout << "INSERTING REDUCTION BUILTIN: " << op->dump() << "\n";
            new_comps.push_back(op);
        }
        new_comps.insert(new_comps.end(), node->decls.begin(), node->decls.end());
        node->decls = new_comps;
        context->exit_module();
    }

    void ReductionOpGenerator::visit(std::shared_ptr<Reduction> node) {
        IRRewriter::visit(node);
        if (reduction_ops.find(node->reductionOp->op->reductionSign) == reduction_ops.end()) {
            reduction_ops.insert({node->reductionOp->op->reductionSign, node->reductionOp});
        }
    }

    void CallRewriter::visit_decl(const std::shared_ptr<FuncDecl> &node) {
        auto new_stmts = std::vector<std::shared_ptr<Statement>>();

        for (auto &stmt: node->body) {
            if (stmt->is_def()) {
                inner_calls.clear();
                temporaries.clear();
                stmt = rewrite(stmt);

                if(inner_calls.empty()) {
                    new_stmts.push_back(stmt);
                    continue;
                }

                std::shared_ptr<Statement> new_stmt;
                for(auto&tensor: temporaries) {
                    auto& call = inner_calls.at(tensor);
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
                            if (def->lhs[i]->tensor->name == "_") {
                                continue;
                            }
                            std::shared_ptr<TensorType> indexed_tensor_type;
                            if (auto t_ = std::dynamic_pointer_cast<Datatype>(t->type->tuple[i])) {
                                indexed_tensor_type = IR::make<TensorType>(t_, std::vector<std::shared_ptr<Expression>>());
                            } else {
                                indexed_tensor_type = std::dynamic_pointer_cast<TensorType>(t->type->tuple[i]);
                            }
//
                            auto indexed_tensor = IR::make<TensorVar>(t->name + "_" + std::to_string(i), indexed_tensor_type, false);
                            auto init = IR::make<Initialize>(indexed_tensor);
                            new_stmts.push_back(init);

                            auto index_assign = IR::make<Definition>(IR::make<Access>(indexed_tensor, std::vector<std::shared_ptr<Expression>>(), std::vector<std::shared_ptr<IndexVarExpr>>()), IR::make<TupleVarReadAccess>(t, i));
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
                        auto acc = IR::make<Access>(t, std::vector<std::shared_ptr<Expression>>(), std::vector<std::shared_ptr<IndexVarExpr>>());
                        new_stmt = IR::make<Definition>(acc, call);
                        new_stmts.push_back(init);
                        new_stmts.push_back(alloc);
                        std::cout << "ADDITIONAL: " << init->dump() << "\n";
                        std::cout << "ADDITIONAL: " << alloc->dump() << "\n";
                        std::cout << "ADDITIONAL: " << new_stmt->dump() << "\n";
                        new_stmts.push_back(new_stmt);
                    }
                }
                if (!std::dynamic_pointer_cast<TupleVar>(stmt->as_def()->rhs)) {
                    new_stmts.push_back(stmt);
                }

            } else {
                new_stmts.push_back(IRRewriter::visit(stmt));
            }
        }

        node->body = new_stmts;
        node_ = node;
    }

    void CallRewriter::visit_call(std::shared_ptr<Call> node) {
        if (!node->getIndices().empty()) {
            call_id += 1;
            node_ = node;
            return;
        }
        for (auto& arg: node->arguments) {
            arg = rewrite(arg);
        }
        std::cout << "VISIT CALL: " << node->dump() << "\n";
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
            temporaries.push_back(tensor);
            node_ = IR::make<ReadAccess>(tensor, std::vector<std::shared_ptr<Expression>>());
            std::cout << "Replacing call node: " << node->dump() << "\n";
            std::cout << "with: " << node_->dump() << "\n";
        } else {
            auto output = IR::make<TupleVar>(name, out_type);
            temporaries.push_back(output);
            inner_calls.insert({output, node});
            node_ = output;
            std::cout << "Replacing call node: " << node->dump() << "\n";
            std::cout << "with: " << node_->dump() << "\n";
        }
        call_id += 1;
    }

    void CallStarConditionProcessor::visit(std::shared_ptr<CallStarCondition> node) {
        std::string name = "call_condition_" + std::to_string(call_id);
        auto one = IR::make<Literal>(1, IR::make<Datatype>(Datatype::Kind::Int));
        auto type = IR::make<TensorType>(IR::make<Datatype>(Datatype::Kind::Bool), std::vector<std::shared_ptr<einsum::Expression>>({one}));
        auto tensor = IR::make<TensorVar>(name, type, false);
        auto idx = IR::make<IndexVar>("i");
        auto acc = IR::make<Access>(tensor, std::vector<std::shared_ptr<Expression>>({IR::make<IndexVarExpr>(idx)}), std::vector<std::shared_ptr<IndexVarExpr>>({IR::make<IndexVarExpr>(idx)}));
        auto accs = std::vector<std::shared_ptr<Access>>({acc});
        reductions.emplace_back();
        inside_stop_condition = true;
        index_vars.clear();
        node->stopCondition = rewrite(node->stopCondition);
        auto& reds = reductions.back();
        node->condition_def = IR::make<Definition>(accs, node->stopCondition, reds);
        reductions.pop_back();
        inside_stop_condition = false;
        node_ = node;
        condition_tensors.insert(tensor);
        visit_call(node);
    }

    void CallStarConditionProcessor::visit_call(std::shared_ptr<Call> node) {
        for(auto& arg: node->arguments) {
            arg = rewrite(arg);
        }
        node_ = node;
        call_id += 1;
    }

    void CallStarConditionProcessor::visit_decl(const std::shared_ptr<FuncDecl>& node) {
        for(auto& s: node->body) {
            s = rewrite(s);
        }
        auto new_stmts = std::vector<std::shared_ptr<Statement>>();

        for(auto& t: condition_tensors) {
            new_stmts.push_back(IR::make<Initialize>(t));
            new_stmts.push_back(IR::make<Allocate>(t));
        }
        auto& old_stmts = node->body;
        new_stmts.insert(new_stmts.end(), old_stmts.begin(), old_stmts.end());
        node->body = new_stmts;
        node_ = node;
    }

    void CallStarConditionProcessor::visit(std::shared_ptr<ReadAccess> node) {
        // must be syntactic sugar for comparing an entire tensor to a scalar
        // by this point all calls should have been taken out in temporaries
        if (node->indices.size() < node->tensor->getOrder() && inside_stop_condition) {
            auto indices = node->indices;
            for(size_t i=node->indices.size(); i < node->tensor->getOrder(); i++) {
                auto idx_var = "i"+ std::to_string(i);
                std::shared_ptr<IndexVar> var = IR::make<IndexVar>(idx_var);
                indices.push_back(IR::make<IndexVarExpr>(var));
                if (index_vars.find(idx_var) == index_vars.end()) {
                    auto& reds = reductions.back();
                    reds.push_back(Reduction::andReduction(var));
                    index_vars.insert(idx_var);
                }
            }
            node->indices = indices;
        }

        node_ = node;

    }

    void DefinitionSplitter::visit(std::shared_ptr<Definition> node) {
        node->rhs->accept(this);
        node->id = def_id;
        node_ = node;
        def_id += 1;
    }

    void DefinitionSplitter::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->rhs->accept(this);
        node->id = def_id;
        node_ = node;
        def_id += 1;
    }

    void MemoryReuseRewriter::visit(std::shared_ptr<Definition> node) {
        tensor_outputs.clear();
        if (std::dynamic_pointer_cast<Call>(node->rhs) && !std::dynamic_pointer_cast<CallStarRepeat>(node->rhs) && !std::dynamic_pointer_cast<CallStarCondition>(node->rhs)) {
//            printf("CALL PN RHS: %$s\n", node->rhs->dump().c_str());
            for(size_t i=0; i < node->lhs.size(); i++) {
                auto& acc = node->lhs[i];
                if (acc->tensor->getOrder() > 0 || acc->tensor->name == "_") {
                    tensor_outputs.push_back(acc->tensor);
                }
            }
            node->rhs->accept(this);
            node_ = node;
        } else {
            IRRewriter::visit(node);
        }
    }

    void MemoryReuseRewriter::visit(std::shared_ptr<Call> node) {
        if (!node->function->is_builtin()) {
            for (size_t i=0; i < tensor_outputs.size(); i++) {
                auto& t = tensor_outputs[i];
                if (t->name == "_") {
                    if (node->function->outputs[i]->getOrder() == 0) {
                        printf("NOT ADDING PLACEHOLDER FOR SCALAR\n");
                        continue;
                    }
                    printf("ADDING PLACEHOLDER FOR TENSOR\n");
                    t = IR::make<TensorVar>("nullptr", t->type, t->is_global);
                }
                auto acc = IR::make<ReadAccess>(t, std::vector<std::shared_ptr<Expression>>());
                node->arguments.push_back(acc);
            }
        }
        tensor_outputs.clear();
        IRRewriter::visit_call(node);
    }


    void FuncDeclRewriter::visit(std::shared_ptr<Allocate> node) {
        if (context->func_scope() && storages.find(node->tensor->name) != storages.end()) {
            node->storage = storages[node->tensor->name];
        }
        node_ = node;
    }

    void FuncDeclRewriter::visit_decl(const std::shared_ptr<FuncDecl>& node) {
        storages.clear();
        for (auto& out: node->outputs) {
            if (out->getOrder() > 0) {
                auto storage = IR::make<TensorVar>(out->name + "_storage", out->type, out->is_global);
                storages.insert({out->name, storage});
                node->storages.push_back(storage);
            } else {
                node->storages.push_back(nullptr);
            }
        }
        IRRewriter::visit_decl(node);
    }
}

