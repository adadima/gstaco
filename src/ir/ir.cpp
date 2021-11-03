//
// Created by Alexandra Dima on 10/13/21.
//

#include <einsum_taco/ir/ir.h>
#include <string>
#include <iostream>

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
                return std::to_string(this->getValue<bool>());
        }
    }

    std::vector<std::shared_ptr<IndexVar>> Literal::getIndices() {
        return {};
    }

    std::string BinaryOp::dump() const {
        auto left = this->left->dump();
        auto right = this->right->dump();
        if (this->left->precedence > this->precedence) {
            left = "(" + left + ")";
        }
        if (this->right->precedence > this->precedence) {
            right = "(" + right + ")";
        }
        return left + " " + this->op->sign + " " + right;
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

    std::shared_ptr<Type> NotExpr::getType() {
        return std::make_shared<Datatype>(Datatype::Kind::Bool);
    };

    std::string UnaryOp::dump() const {
        return this->op->sign + " " + this->expr->dump();
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
        std::vector<std::shared_ptr<DimensionType>> dims;
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
        auto def = this->lhs->dump() + " = " + this->rhs->dump();
        if (this->reductions.empty()) {
            return def;
        }
        def += " | ";
        bool first = true;
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

        auto def = "Let " + this->funcName + "(" + inParams + ") -> (" + outParams + ")";

        std::string body;
        for (const auto &i : this->body) {
            body += i->dump() + "\n";
        }
        return def + "End";
    }

    std::shared_ptr<Type> Call::getType() {
        return std::make_shared<TupleType>(this->function->getOutputType());
    }

    std::string Call::dump() const {
        std::string args;
        for (int i=0; i < this->arguments.size(); i++) {
            args += this->arguments[i]->dump();
            if (i < this->arguments.size() - 1) {
                args += ", ";
            }
        }
        return this->function->funcName + "(" + args + ")";
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
        auto call = Call::dump();
        return call + " | " + std::to_string(this->numIterations);
    }

    std::string CallStarCondition::dump() const {
        auto call = Call::dump();
        return call + " | (" + this->stopCondition->dump() + ")";
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
}