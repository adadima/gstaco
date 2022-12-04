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
        struct DefinitionScope {
            std::shared_ptr<Definition> def;
            std::set<std::string> leftIndexVars;
            std::set<std::string> reductionIndexVars;
            std::map<std::string, std::set<std::shared_ptr<Expression>>> index_var_dimensions_;

            [[nodiscard]] bool has_index_var(const std::string& i) const {
                return leftIndexVars.count(i) || reductionIndexVars.count(i);
            }

            std::shared_ptr<Expression> getDimension(const std::string& name) {
                if (index_var_dimensions_.count(name)) {
                    auto dims = index_var_dimensions_[name];
                    if (!dims.empty()) {
                        return *dims.begin();
                    }
                }
                return nullptr;
            }

        };

        std::map<std::string, std::shared_ptr<TensorVar>> globals_;
        std::map<std::string, std::shared_ptr<TensorVar>> locals_;
        std::map<std::string, std::shared_ptr<FuncDecl>>  functions_;

        std::shared_ptr<FuncDecl> func_scope_;
        DefinitionScope* def_scope_;
        std::shared_ptr<Call> call_scope_;
        std::shared_ptr<Access> access_scope_;
        std::shared_ptr<ReadAccess> read_access_scope_;
        std::stack<std::shared_ptr<TensorVar>> tensor_scope_;
        std::map<std::string, std::shared_ptr<IndexVar>> reduction_dimensions_;

        int coordinate_;

        static std::shared_ptr<TensorVar> get_param(const std::shared_ptr<TensorVar>& tensor, const std::vector<std::shared_ptr<TensorVar>>& param_list) {
            if (tensor->name.find('#', 0) == 0) {
                return tensor;
            }

            for (auto &&value : param_list) {
                if (value->name == tensor->name) {
                    return value;
                }
                std::cout << "NOT MATCHING PARAM: " << value->name << "\n";
            }
            std::cout << "PARAM: " << tensor->name << "\n";

            return nullptr;
        }

    public:
        std::shared_ptr<FuncDecl>& func_scope() {
            return func_scope_;
        }

        DefinitionScope*& def_scope() {
            return def_scope_;
        }

        std::shared_ptr<Call>& call_scope() {
            return call_scope_;
        }

        std::shared_ptr<Access>& access_scope() {
            return access_scope_;
        }

        std::shared_ptr<ReadAccess>& read_access_scope() {
            return read_access_scope_;
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

        bool is_global(const std::shared_ptr<TensorVar>& tensor) {
            if (tensor->name.find('#', 0) ==0) {
                return false;
            }
            if (func_scope() && (get_param(tensor, func_scope()->inputs) || get_param(tensor, func_scope()->outputs))) {
                return false;
            }
            if (globals_.count(tensor->name) != 0) {
                return true;
            }
            std::abort();
        }

        std::shared_ptr<FuncDecl> get_function(const std::string& name) {
            if (functions_.count(name)) {
                return functions_[name];
            }
            return nullptr;
        }

        void enter_module(const std::shared_ptr<Module>& module) {
            for (auto &global : module->get_globals()) {
                add_global(global);
            }
        }

        void exit_module() {
            globals_.clear();
        }

        void enter_function(const std::shared_ptr<FuncDecl>& func) {
            func_scope() = func;
        }

        void enter_definition(const std::shared_ptr<Definition>& def) {

            auto scope = new DefinitionScope{def, def->getReductionVars(), def->getLeftIndexVars(), def->getIndexVarDims(this)};
            def_scope() = scope;
        }

        void enter_call(const std::shared_ptr<Call>& call) {
            call_scope() = call;
        }

        std::shared_ptr<IndexVar> get_index_var(std::string name) {
            if (tensor_scope().empty()) {
                auto dim = def_scope()->getDimension(name);
                if (dim) {
                    return std::make_shared<IndexVar>(name, dim);
                }
                return nullptr;
            }

            auto tensor = get_write_tensor(tensor_scope().top());
            if (tensor) {
                auto dim = tensor->getType()->getDimension(coordinate_);
                return IR::make<IndexVar>(name, dim);
            }

            return nullptr;
        }

        std::shared_ptr<IndexVarExpr> get_index_var_expr(std::string name) {
            if (tensor_scope().empty()) {
                auto dim = def_scope()->getDimension(name);
                auto index_var = std::make_shared<IndexVar>(name, dim);
                return std::make_shared<IndexVarExpr>(index_var);
            }
            auto tensor = get_read_tensor(tensor_scope().top());
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

        void exit_definition(std::shared_ptr<Definition> definition) {
            reduction_dimensions_.clear();
            for (auto &acc: definition->lhs) {
                locals_.emplace(acc->tensor->name, acc->tensor);
            }
            def_scope() = nullptr;
        }

        void exit_call(std::shared_ptr<Call> call) {
            call_scope() = nullptr;
        }

        void exit_function(const std::shared_ptr<FuncDecl>& func) {
            functions_.emplace(func->funcName, func);
            locals_.clear();
            func_scope() = nullptr;
        }

        std::shared_ptr<TensorVar> get_read_tensor(const std::shared_ptr<TensorVar>& tensor) {
            if (locals_.count(tensor->name) > 0) {
                return locals_[tensor->name];
            }
            auto t = get_param(tensor, func_scope()->inputs);
            if (t != nullptr) {
                return t;
            }
            if (globals_.count(tensor->name) > 0) {
                return globals_[tensor->name];
            }
            return tensor;
        }

        std::shared_ptr<TensorVar> get_write_tensor(const std::shared_ptr<TensorVar>& tensor) {
            auto t = get_param(tensor, func_scope()->outputs);
            if (t == nullptr) {
                return tensor;
            }
            return t;
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
