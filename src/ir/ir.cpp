//
// Created by Alexandra Dima on 10/13/21.
//
#include <einsum_taco/ir/type.h>
#include <einsum_taco/ir/ir.h>
#include <string>
#include <iostream>
//auto obj = std::static_pointer_cast<einsum::Expression>(einsum::IR::make<einsum::Literal>($1, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Int))); $$ = &obj;
namespace einsum {

    bool Literal::isInt() const {
        return this->getDatatype()->isInt();
    }

    bool Literal::isFloat() const {
        return this->getDatatype()->isFloat();
    }

    bool Literal::isBool() const {
        return this->getDatatype()->isBool();
    }

    std::shared_ptr<Type> Literal::getType() {
        return type;
    };

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
//        std::cout << "Precedence: " << this->precedence << "\n";
//        std::cout << "Left precedence: " << this->left->precedence << "\n";
//        std::cout << "Right precedence: " << this->right->precedence << "\n";
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

    std::shared_ptr<Type> ArithmeticExpression::getType() {
        if (this->left->getType()->isFloat() || this->right->getType()->isFloat()) {
            return std::make_shared<Datatype>(Datatype::Kind::Float);
        }
        return std::make_shared<Datatype>(Datatype::Kind::Int);
    };

    std::shared_ptr<Type> ModuloExpression::getType() {
        return std::make_shared<Datatype>(Datatype::Kind::Int);
    };

    std::shared_ptr<Type> LogicalExpression::getType() {
        return std::make_shared<Datatype>(Datatype::Kind::Bool);
    };

    std::shared_ptr<Type> ComparisonExpression::getType() {
        return std::make_shared<Datatype>(Datatype::Kind::Bool);
    };

    std::shared_ptr<Type> NotExpression::getType() {
        return std::make_shared<Datatype>(Datatype::Kind::Bool);
    };

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

    std::shared_ptr<Type> IndexVarExpr::getType() {
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

    std::shared_ptr<Type> ReadAccess::getType() {
        std::vector<std::shared_ptr<Expression>> dims;
        unsigned long last = this->tensor->type->getOrder() - this->indices.size();
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
        if (this->reductions.empty()) {
            return def;
        }
        def += " | ";
        first = true;
        for (std::pair<std::shared_ptr<IndexVar>, std::shared_ptr<Reduction>> element : this->reductions) {
            if (!first) {
                def += ", ";
            } else {
                first = false;
            }
            def += element.second->dump();
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

    std::shared_ptr<Type> Call::getType() {
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
        auto mapper = [](const std::shared_ptr<TensorVar> tvar) { return tvar->type; };
        std::transform(this->outputs.begin(), this->outputs.end(), std::back_inserter(types), mapper);
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

    FuncDecl& ModuleComponent::as_decl() {
        return dynamic_cast<FuncDecl&>(*this);
    }

    const FuncDecl& ModuleComponent::as_decl() const {
        return dynamic_cast<const FuncDecl&>(*this);
    }

    bool ModuleComponent::is_def() const {
        return dynamic_cast<const Definition*>(this) != nullptr;
    }

    Definition& ModuleComponent::as_def() {
        return dynamic_cast<Definition&>(*this);
    }

    const Definition& ModuleComponent::as_def() const {
        return dynamic_cast<const Definition&>(*this);
    }

    bool ModuleComponent::is_expr() const {
        return dynamic_cast<const Expression*>(this) != nullptr;
    }

    Expression& ModuleComponent::as_expr() {
        return dynamic_cast<Expression&>(*this);
    }

    const Expression& ModuleComponent::as_expr() const {
        return dynamic_cast<const Expression&>(*this);
    }
}