//
// Created by Alexandra Dima on 26.12.2021.
//

#ifndef EINSUM_TACO_CONTEXT_H
#define EINSUM_TACO_CONTEXT_H

#include "einsum_taco/ir/ir.h"
#include <iostream>
#include <stack>

namespace einsum {
    class IRContext {

        std::map<std::string, std::shared_ptr<TensorVar>> globals_;
        std::map<std::string, std::shared_ptr<FuncDecl>>  functions_;

        std::shared_ptr<FuncDecl> func_scope_;
        std::shared_ptr<Definition> def_scope_;
        std::shared_ptr<Access> access_scope_;
        std::stack<std::shared_ptr<TensorVar>> tensor_scope_;
        std::map<std::string, std::shared_ptr<IndexVar>> reduction_dimensions_;

        int coordinate_;

        std::map<std::string, std::shared_ptr<Expression>> index_var_dimensions_;

        static std::shared_ptr<TensorVar> get_param(const TensorVar& tensor, const std::vector<std::shared_ptr<TensorVar>>& param_list) {


            for (auto &&value : param_list) {

                if (value->name == tensor.name) {
                    return value;
                }
            }

            return nullptr;
        }

    public:
        std::shared_ptr<FuncDecl>& func_scope() {
            return func_scope_;
        }

        std::shared_ptr<Definition>& def_scope() {
            return def_scope_;
        }

        std::shared_ptr<Access>& access_scope() {
            return access_scope_;
        }

        std::stack<std::shared_ptr<TensorVar>>& tensor_scope() {
            return tensor_scope_;
        }

        int& coordinate() {
            return coordinate_;
        }

        void add_global(const std::shared_ptr<TensorVar>& tensor) {
            globals_.emplace(tensor->name, tensor);
        }

        void enter_function(const std::shared_ptr<FuncDecl>& func) {
            functions_.emplace(func->funcName, func);

            func_scope() = func;


        }

        void enter_definition(const std::shared_ptr<Definition>& def) {

            def_scope() = def;

            if (!func_scope()) {
                for(auto &&acc : def->lhs) {
                    add_global(acc->tensor);
                }
            }
        }

        std::shared_ptr<IndexVar> get_index_var(std::string name) {

            auto tensor = get_write_tensor(*tensor_scope().top());
            if (tensor) {
                auto dim = tensor->getType()->getDimension(coordinate_);
                return std::make_shared<IndexVar>(name, dim);
            }
            return nullptr;
        }

        std::shared_ptr<IndexVarExpr> get_index_var_expr(std::string name) {
            auto tensor = get_read_tensor(*tensor_scope().top());
            if (tensor) {
                auto dim = tensor->getType()->getDimension(coordinate_);
                auto index_var = std::make_shared<IndexVar>(name, dim);
                return std::make_shared<IndexVarExpr>(index_var);
            }
            return nullptr;
        }

        void add_reduction_var(const std::shared_ptr<IndexVar>& ivar) {
            reduction_dimensions_.emplace(ivar->getName(), ivar);
        }

        std::shared_ptr<IndexVar> get_reduction_var(const std::string& name) {
            if (reduction_dimensions_.count(name)) {
                return reduction_dimensions_[name];
            }
            return nullptr;
        }

        void exit_definition() {
            reduction_dimensions_.clear();
            def_scope() = nullptr;
        }

        void exit_function() {
            func_scope() = nullptr;
        }

        std::shared_ptr<TensorVar> get_read_tensor(const TensorVar& tensor) {
            if (globals_.count(tensor.name)) {
                return globals_[tensor.name];
            }
            return get_param(tensor, func_scope()->inputs);
        }

        std::shared_ptr<TensorVar> get_write_tensor(const TensorVar& tensor) {
            return get_param(tensor, func_scope()->outputs);
        }

        void enter_access(const std::shared_ptr<Access>& access) {

            access_scope() = access;
            tensor_scope().push(access->tensor);
            coordinate() = -1;
        }

        void exit_access() {
            access_scope() = nullptr;
            tensor_scope().pop();
        }

        void enter_read_access(const std::shared_ptr<ReadAccess>& raccess) {
            tensor_scope().push(raccess->tensor);
            coordinate() = -1;

        }

        void exit_read_access() {
            tensor_scope().pop();
        }

        void advance_access() {
            coordinate()++;
        }




    };
}


#endif //EINSUM_TACO_CONTEXT_H
