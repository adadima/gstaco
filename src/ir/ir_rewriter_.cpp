//
// Created by Alexandra Dima on 07.01.2022.
//
#include "einsum_taco/ir/ir_rewriter_.h"
#include "einsum_taco/base/assert.h"
#include <iostream>

// TODO : rewrite IR to break up definitions of multiple inputs

namespace einsum {

    template<typename T>
    std::shared_ptr<T> shared_from_ref(T& ref) {
        return std::dynamic_pointer_cast<T>(ref.shared_from_this());
    }

    void IndexVarRewriter::visit(IndexVar& node) {
        index_var = context->get_index_var(node.getName());
        context->add_reduction_var(index_var);
    }

    void  IndexVarRewriter::visit(Literal& node) {
        expr = shared_from_ref(node);
    }

    void  IndexVarRewriter::visit(TensorVar& node) {

        if (context->access_scope()) {
            tensor = context->get_write_tensor(node);
        } else {
            tensor = context->get_read_tensor(node);
        }
    }

    void  IndexVarRewriter::visit(IndexVarExpr& node) {
        auto ivar = context->get_index_var_expr(node.getName());
        context->add_reduction_var(ivar->indexVar);
        expr = ivar;
    }

    void  IndexVarRewriter::visit(Access& node) {
        context->enter_access(shared_from_ref(node));

        node.tensor->accept(this);
        auto new_tensor = tensor;

        auto indices = std::vector<std::shared_ptr<IndexVar>>();

        for (auto &&indice : node.indices) {
            context->advance_access();
            indice->accept(this);
            indices.push_back(index_var);
        }

        access = std::make_shared<Access>(new_tensor, indices);

        context->exit_access();
    }

    void  IndexVarRewriter::visit(ReadAccess& node) {
        if (node.indices.empty() && context->def_scope()->has_index_var(node.tensor->name)) {
            auto ivar = context->get_index_var_expr(node.tensor->name);
            context->add_reduction_var(ivar->indexVar);
            expr = ivar;
            return;
        }
        context->enter_read_access(shared_from_ref(node));
        node.tensor->accept(this);

        auto new_tensor = tensor;

        auto new_indices = std::vector<std::shared_ptr<Expression>>();
        for (auto & indice : node.indices) {
            context->advance_access();
            indice->accept(this);
            new_indices.push_back(expr);
        }

        expr = std::make_shared<ReadAccess>(new_tensor, new_indices);
        context->exit_read_access();
    }

    void  IndexVarRewriter::visit(Definition& node) {
        if (!context->func_scope()) {
            return;
        }

        context->enter_definition(shared_from_ref(node));

        auto new_lhs = std::vector<std::shared_ptr<Access>>();
        for (auto &&acc : node.lhs) {
            acc->accept(this);
            new_lhs.push_back(access);
        }

        node.rhs->accept(this);
        auto new_rhs = expr;

        auto new_reds = std::vector<std::shared_ptr<Reduction>>();

        for (auto &red : node.reduction_list) {
            auto new_red = IR::make<Reduction>(context->get_reduction_var(red->reductionVar->getName()), red->reductionOp, red->reductionInit);
            new_reds.push_back(new_red);
        }

        def = IR::make<Definition>(new_lhs, new_rhs, new_reds);

        context->exit_definition();
    }

    void  IndexVarRewriter::visit(FuncDecl& node) {
        context->enter_function(shared_from_ref(node));

        auto body = std::vector<std::shared_ptr<Definition>>();
        for (auto &&definition : node.body) {
            definition->accept(this);
            body.push_back(def);
        }

        func = std::make_shared<FuncDecl>(node.funcName, node.inputs, node.outputs, body);

        context->exit_function(func);

    }

    void  IndexVarRewriter::visit(Call& node) {
        auto new_args = std::vector<std::shared_ptr<Expression>>();

        for (auto &&arg : node.arguments) {
            arg->accept(this);
            new_args.push_back(expr);
        }
        auto function = context->get_function(node.function->funcName);
        expr = std::make_shared<Call>(function, new_args);
    }

    void  IndexVarRewriter::visit(CallStarRepeat& node) {
        auto new_args = std::vector<std::shared_ptr<Expression>>();

        for (auto &&arg : node.arguments) {
            arg->accept(this);
            new_args.push_back(expr);
        }

        auto function = context->get_function(node.function->funcName);
        expr = std::make_shared<CallStarRepeat>(node.numIterations, function, new_args);

    }

    void  IndexVarRewriter::visit(CallStarCondition& node) {
        auto new_args = std::vector<std::shared_ptr<Expression>>();

        for (auto &&arg : node.arguments) {
            arg->accept(this);
            new_args.push_back(expr);
        }

        auto new_func = context->get_function(node.function->funcName);

        node.stopCondition->accept(this);
        auto new_stop = expr;

        auto cond = std::dynamic_pointer_cast<Literal>(new_stop);
        if (cond && cond->isInt()) {
            expr = IR::make<CallStarRepeat>(cond->getValue<int>(), new_func, node.arguments);
        } else {
            expr = IR::make<CallStarCondition>(new_stop, new_func, new_args);
        }
    }

    //TODO: remove this method and find a cleaner way to retrieve return type of visited module component
    std::shared_ptr<ModuleComponent> IndexVarRewriter::visit(ModuleComponent& node) {
        node.accept(this);
        if (node.is_decl()) {
            return func;
        }
        if (node.is_def()) {
            return def;
        }

        if (node.is_expr()) {
            return expr;
        }

        return nullptr;
    }

    void  IndexVarRewriter::visit(Module& node) {
        auto new_components = std::vector<std::shared_ptr<ModuleComponent>>();
        for (auto &&comp : node.decls) {
            new_components.push_back(visit(*comp));
        }
        module = std::make_shared<Module>(new_components);
    }

    void IndexVarRewriter::visit(Reduction &node) {}

    void IndexVarRewriter::visit(BinaryOp &node) {
        node.left->accept(this);
        auto new_left = expr;

        node.right->accept(this);
        auto new_right = expr;

        expr = std::make_shared<BinaryOp>(new_left, new_right, node.op, node.getType());
    }

    void IndexVarRewriter::visit(UnaryOp &node) {
        node.expr->accept(this);
        auto exp = expr;
        expr = std::make_shared<UnaryOp>(exp, node.op, node.getType());
    }
}