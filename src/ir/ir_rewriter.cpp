//
// Created by Alexandra Dima on 07.01.2022.
//
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/base/assert.h"
#include <iostream>

namespace einsum {

    template<typename T>
    std::shared_ptr<T> shared_from_ref(T& ref) {
        return std::dynamic_pointer_cast<T>(ref.shared_from_this());
    }

    std::shared_ptr<BinaryOp> IRRewriter::rewrite_binary(BinaryOp& node) {
        node.left->accept(this);
        auto new_left = expr;

        node.right->accept(this);
        auto new_right = expr;

        auto bin = std::make_shared<BinaryOp>(new_left, new_right, node.op, node.getType());

        return bin;
    }

    std::shared_ptr<UnaryOp> IRRewriter::rewrite_unary(UnaryOp& node) {
        node.expr->accept(this);
        auto exp = expr;
        return std::make_shared<UnaryOp>(exp, node.op, node.getType());
    }

    void IRRewriter::visit(IndexVar& node) {
        index_var = context->get_index_var(node.getName());
        context->add_reduction_var(index_var);
    }

    void  IRRewriter::visit(Literal& node) {
        expr = shared_from_ref(node);
    }

    void  IRRewriter::visit(TensorVar& node) {

        if (context->access_scope()) {
            tensor = context->get_write_tensor(node);
        } else {
            tensor = context->get_read_tensor(node);
        }
    }

    void  IRRewriter::visit(IndexVarExpr& node) {
        auto ivar = context->get_index_var_expr(node.getName());
        context->add_reduction_var(ivar->indexVar);
        expr = ivar;
    }

    void  IRRewriter::visit(Access& node) {
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

    void  IRRewriter::visit(ReadAccess& node) {
        if (!context->tensor_scope().empty() && node.indices.empty() && context->def_scope()->has_index_var(node.tensor->name)) {
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

    void  IRRewriter::visit(Definition& node) {
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

        for (auto &[ivar, red] : node.reductions) {
                auto new_red = IR::make<Reduction>(context->get_reduction_var(ivar->getName()), red->reductionOp, red->reductionInit);
                new_reds.push_back(new_red);
        }

        def = IR::make<Definition>(new_lhs, new_rhs, new_reds);

        context->exit_definition();
    }

    void  IRRewriter::visit(FuncDecl& node) {
        context->enter_function(shared_from_ref(node));

        auto body = std::vector<std::shared_ptr<Definition>>();
        for (auto &&definition : node.body) {
            definition->accept(this);
            body.push_back(def);
        }

        func = std::make_shared<FuncDecl>(node.funcName, node.inputs, node.outputs, body);

        context->exit_function();

    }

    void  IRRewriter::visit(Call& node) {
        auto new_args = std::vector<std::shared_ptr<Expression>>();

        for (auto &&arg : node.arguments) {
            arg->accept(this);
            new_args.push_back(expr);
        }

        node.function->accept(this);
        expr = std::make_shared<Call>(func, new_args);
    }

    void  IRRewriter::visit(CallStarRepeat& node) {
        auto new_args = std::vector<std::shared_ptr<Expression>>();

        for (auto &&arg : node.arguments) {
            arg->accept(this);
            new_args.push_back(expr);
        }

        node.function->accept(this);
        expr = std::make_shared<CallStarRepeat>(node.numIterations, func, new_args);

    }

    void  IRRewriter::visit(CallStarCondition& node) {
        auto new_args = std::vector<std::shared_ptr<Expression>>();

        for (auto &&arg : node.arguments) {
            arg->accept(this);
            new_args.push_back(expr);
        }

        node.function->accept(this);
        auto new_func = func;

        node.stopCondition->accept(this);
        auto new_stop = expr;

        expr = std::make_shared<CallStarCondition>(new_stop, new_func, new_args);
    }

    //TODO: remove this method and find a cleaner way to retrieve return type of visited module component
    std::shared_ptr<ModuleComponent> IRRewriter::visit(ModuleComponent& node) {
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

    void  IRRewriter::visit(Module& node) {
        auto new_components = std::vector<std::shared_ptr<ModuleComponent>>();
        for (auto &&comp : node.decls) {
            new_components.push_back(visit(*comp));
        }
        module = std::make_shared<Module>(new_components);
    }

    void IRRewriter::visit(Reduction &node) {}

    void IRRewriter::visit(BinaryOp &node) {
        expr = rewrite_binary(node);
    }

    void IRRewriter::visit(UnaryOp &node) {
        expr = rewrite_unary(node);
    }

}