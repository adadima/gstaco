//
// Created by Alexandra Dima on 10/13/21.
//
#include <einsum_taco/ir/type.h>
#include <einsum_taco/ir/ir.h>
#include <string>
#include <iostream>
#include <einsum_taco/ir/context.h>
#include <set>
#include <map>
#include <cxxabi.h>
#include <algorithm>


namespace einsum {

    std::string IR::class_name() const {
        int status;
        char* demangled_name_buf = abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, &status);
        auto demangled_name = std::string{demangled_name_buf};
        free(demangled_name_buf);
        if (demangled_name.find("einsum::") == 0) {
            return demangled_name.substr(8);
        }
        return demangled_name;
    }


    bool Literal::isInt() const {
        return this->getDatatype()->isInt();
    }

    bool Literal::isFloat() const {
        return this->getDatatype()->isFloat();
    }

    bool Literal::isBool() const {
        return this->getDatatype()->isBool();
    }

    std::shared_ptr<Type> Literal::getType() const {
        return type;
    }

    std::string Literal::dump() const {
        switch (this->getDatatype()->getKind()) {
            case Datatype::Kind::Int:
                return std::to_string(this->getValue<int>());
            case Datatype::Kind::Float:
                return std::to_string(this->getValue<float>());
            case Datatype::Kind::Bool:
                return this->getValue<bool>() ? "true" : "false";
//                return std::to_string(this->getValue<bool>());
        }
    }

    std::vector<std::shared_ptr<IndexVar>> Literal::getIndices() {
        return {};
    }

    std::string BinaryOp::dump() const {
        auto left_ = this->left->dump();
        auto right_ = this->right->dump();
        if (this->left->precedence > this->precedence) {
            left_ = "(" + left_ + ")";
        }
        if ((this->right->precedence > this->precedence) ||  (this->right->precedence == this->precedence && this->isAsymmetric)){
            right_ = "(" + right_ + ")";
        }
        return left_ + " " + this->op->sign + " " + right_;
    }

    std::vector<std::shared_ptr<IndexVar>> BinaryOp::getIndices() {
        std::vector<std::shared_ptr<IndexVar>> lIndices = this->left->getIndices();
        std::vector<std::shared_ptr<IndexVar>> rIndices = this->right->getIndices();
        std::vector<std::shared_ptr<IndexVar>> inds(lIndices);
        inds.insert(inds.end(), lIndices.begin(), lIndices.end());
        inds.insert(inds.end(), rIndices.begin(), rIndices.end());
        return inds;
    }

    std::string UnaryOp::dump() const {
        return this->op->sign + " " + this->expr->dump();
    }

    std::vector<std::shared_ptr<IndexVar>> UnaryOp::getIndices() {
        return this->expr->getIndices();
    }

    std::string TensorVar::dump() const {
        return this->name;
    }

    std::shared_ptr<TensorType> TensorVar::getType() const {
        return this->type;
    }

    std::string IndexVar::dump() const {
        return this->name;
    }

    std::string IndexVarExpr::dump() const {
        return this->indexVar->name;
    }

    std::vector<std::shared_ptr<IndexVar>> IndexVarExpr::getIndices() {
        return {this->indexVar};
    }

    std::shared_ptr<Type> IndexVarExpr::getType() const {
        return std::make_shared<Datatype>(Datatype::Kind::Int);
    }

    std::string Access::dump() const {
        auto acc = this->tensor->name;
        for (const auto &indice : this->indices) {
            acc += "[" + indice->dump() + "]";
        }
        return acc;
    }

    std::string ReadAccess::dump() const {
        auto acc = this->tensor->name;
        for (const auto &indice : this->indices) {
            acc += "[" + indice->dump() + "]";
        }
        return acc;
    }

    std::vector<std::shared_ptr<IndexVar>> ReadAccess::getIndices() {
        std::vector<std::shared_ptr<IndexVar>> indices;
        for (const auto &indExpr : this->indices) {
            auto inds = indExpr->getIndices();
            indices.insert(indices.end(), inds.begin(), inds.end());
        }
        return indices;
    }

    std::shared_ptr<Type> ReadAccess::getType() const {
        std::vector<std::shared_ptr<Expression>> dims;
        int last = this->tensor->type->getOrder() - this->indices.size();
        if (last < 0) {
            last = 0;
        }
        dims.reserve(last);
        for (int i=0; i < last; i++) {
            dims.push_back(this->tensor->type->getDimension(i));
        }
        if (dims.empty()) {
            return this->tensor->type->getElementType();
        }
        return std::make_shared<TensorType>(this->tensor->type->getElementType(), dims);
    }

    std::string Reduction::dump() const {
        return this->reductionVar->dump() + ":" + "(" + this->reductionOp->reductionSign + ", " + this->reductionInit->dump() + ")";
    }

    std::string Definition::dump() const {
        std::string def; // = this->lhs->dump() + " = " + this->rhs->dump();
        bool first = true;
        for (const auto & lh : this->lhs) {
            if (!first) {
                def += ", ";
            } else {
                first = false;
            }
            def += lh->dump();
        }

        def += " = " + this->rhs->dump();
        if (this->reduction_list.empty()) {
            return def;
        }
        def += " | ";
        first = true;
        for (auto& red: this->reduction_list) {
            if (!first) {
                def += ", ";
            } else {
                first = false;
            }
            def += red->dump();
        }
        return def;
    }

    std::string FuncDecl::dump() const {
        std::string inParams;
        auto inputs = this->inputs;
        auto outputs = this->outputs;
        auto inputTypes = this->getInputType();
        auto outputTypes = this->getOutputType();

        for (int i=0; i < inputs.size(); i++) {
            inParams += (inputs[i]->dump() + " " + inputTypes[i]->dump());
            if (i < inputs.size() - 1) {
                inParams += ", ";
            }
        }

        std::string outParams;
        for (int i=0; i < outputs.size(); i++) {
            outParams += (outputs[i]->dump() + " " + outputTypes[i]->dump());
            if (i < outputs.size() - 1) {
                outParams += ", ";
            }
        }

        auto def = "Let " + this->funcName + "(" + inParams + ") -> (" + outParams + ")\n";

        std::string body;
        for (const auto &i : this->body) {
            body += "    " + i->dump() + "\n";
        }
        return def + body + "End";
    }

    std::shared_ptr<Type> Call::getType() const {
        return std::make_shared<TupleType>(this->function->getOutputType());
    }

    std::string Call::dump_args() const {
        std::string args;
        for (int i=0; i < this->arguments.size(); i++) {
            args += this->arguments[i]->dump();
            if (i < this->arguments.size() - 1) {
                args += ", ";
            }
        }
        return "(" + args + ")";
    }

    std::string Call::dump() const {
        return this->function->funcName + this->dump_args();
    }

    std::vector<std::shared_ptr<IndexVar>> Call::getIndices() {
        std::vector<std::shared_ptr<IndexVar>> inds;
        for (auto &argument : this->arguments) {
            auto newInds = argument->getIndices();
            inds.insert(inds.end(), newInds.begin(), newInds.end());
        }
        return inds;
    }

    std::string CallStarRepeat::dump() const {
        auto call = this->function->funcName + "*" + this->dump_args();
        return call + " | " + std::to_string(this->numIterations);
    }

    std::string CallStarCondition::dump() const {
        auto call = this->function->funcName + "*" + this->dump_args();
        //  TODO: think of when to wrap condition around parens
        return call + " | " + "(" + this->stopCondition->dump() + ")";
    }

    std::vector<std::shared_ptr<Type>> FuncDecl::getInputType() const {
        std::vector<std::shared_ptr<Type>> types;
        auto mapper = [](const std::shared_ptr<TensorVar> tvar) { return tvar->type; };
        std::transform(this->inputs.begin(), this->inputs.end(), std::back_inserter(types), mapper);
        return types;
    }

    std::vector<std::shared_ptr<Type>> FuncDecl::getOutputType() const {
        std::vector<std::shared_ptr<Type>> types;
        for (auto &var : outputs) {
            types.push_back(var->getType());
        }
        return types;
    }

    std::string Module::dump() const {
        std::string code;
        for (const auto &d: decls) {
            code += "\n";
            code += d->dump();
            code += "\n";
        }
        return code;
    }

    void Module::add(std::shared_ptr<ModuleComponent> decl) {
        decls.push_back(std::move(decl));
    }

    bool ModuleComponent::is_decl() const {
        return dynamic_cast<const FuncDecl*>(this) != nullptr;
    }

    std::shared_ptr<FuncDecl> ModuleComponent::as_decl() {
        try {
            return std::dynamic_pointer_cast<FuncDecl>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_def() const {
        return dynamic_cast<const Definition*>(this) != nullptr;
    }

    std::shared_ptr<Definition> ModuleComponent::as_def() {
        try {
            return std::dynamic_pointer_cast<Definition>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_expr() const {
        return dynamic_cast<const Expression*>(this) != nullptr;
    }

    std::shared_ptr<Expression> ModuleComponent::as_expr() {
        try {
            return std::dynamic_pointer_cast<Expression>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    std::string IndexVar::getName() const {
        return name;
    }

    std::shared_ptr<Expression> IndexVar::getDimension(int i) const {
        return dimension;
    }

    std::string IndexVarExpr::getName() const {
        return indexVar->getName();
    }

    std::shared_ptr<Expression> IndexVarExpr::getDimension(int i) const {
        return indexVar->getDimension(i);
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  Literal::getIndexVarDims(IRContext* context) const {
        return {};
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  BinaryOp::getIndexVarDims(IRContext* context) const {
        auto left_dims = left->getIndexVarDims(context);
        auto right_dims = right->getIndexVarDims(context);

        for (const auto &[key, value] : right_dims) {
            left_dims.emplace(key, value);
        }
        return left_dims;
    }

    std::shared_ptr<Type> BinaryOp::getType() const {
        return type;
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  UnaryOp::getIndexVarDims(IRContext* context) const {
        return expr->getIndexVarDims(context);
    }

    std::shared_ptr<Type> UnaryOp::getType() const {
        return type;
    }

    template<class T, class E>
    std::map<std::string, std::set<std::shared_ptr<Expression>>>  getDimsFromAccess(const std::shared_ptr<const TensorVar>& tensor, const std::vector<std::shared_ptr<E>>& indices) {
        auto dims = std::map<std::string, std::set<std::shared_ptr<Expression>>> ();

        for (int i=0; i < indices.size(); i++) {
            auto ind = indices[i];
            auto index_var = std::dynamic_pointer_cast<T>(ind);
            if (index_var) {
                auto dimension = tensor->type->getDimension(i);
                dims[index_var->getName()].insert(dimension);
            }
        }

        return dims;
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  ReadAccess::getIndexVarDims(IRContext* context) const {
        return getDimsFromAccess<IndexVarExpr, Expression>(context->get_read_tensor(tensor), indices);
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  IndexVarExpr::getIndexVarDims(IRContext* context) const {
        return {};
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  Call::getIndexVarDims(IRContext* context) const {
        auto dims = std::map<std::string, std::set<std::shared_ptr<Expression>>> ();

        for (auto &&arg : arguments) {
            for (const auto &[key, value] : arg->getIndexVarDims(context)) {
                auto &entry = dims[key];
                entry.insert(value.begin(), value.end());
            }
        }

        return dims;
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  Access::getIndexVarDims(IRContext* context) const {
        return getDimsFromAccess<IndexVar, IndexVar>(context->get_write_tensor(tensor), indices);
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>> Definition::getIndexVarDims(IRContext* context) const {
        auto dims = std::map<std::string, std::set<std::shared_ptr<Expression>>> ();

        for (auto &&acc : lhs) {
            for (const auto &[key, value] : acc->getIndexVarDims(context)) {
                auto &entry = dims[key];
                entry.insert(value.begin(), value.end());
            }
        }

        for (const auto &[key, value] : rhs->getIndexVarDims(context)) {
            auto &entry = dims[key];
            entry.insert(value.begin(), value.end());
        }

        return dims;
    }

    std::set<std::string> Definition::getLeftIndexVars() const {
        auto s = std::set<std::string>();
        for (auto &acc: lhs) {
            for(auto &idx: acc->indices) {
                s.insert(idx->getName());
            }
        }
        return s;
    }

    std::set<std::string> Definition::getReductionVars() const {
        auto s = std::set<std::string>();
        for (auto &&var: reduction_list) {
            s.insert(var->reductionVar->getName());
        }
        return s;
    }


}