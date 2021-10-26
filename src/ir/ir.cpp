//
// Created by Alexandra Dima on 10/13/21.
//

#include <einsum_taco/ir/ir.h>
#include <string>

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

    TensorType* TensorVar::getType() const {
        return this->type;
    }

    Type* FuncDecl::getInputType() const {
        std::vector<Type*> types;
        auto mapper = [](const TensorVar& tvar) { return tvar.type; };
        std::transform(this->inputs.begin(), this->inputs.end(), std::back_inserter(types), mapper);
        return new TupleType(types);
    }

    Type* FuncDecl::getOutputType() const {
        std::vector<Type*> types;
        auto mapper = [](const TensorVar& tvar) { return tvar.type; };
        std::transform(this->outputs.begin(), this->outputs.end(), std::back_inserter(types), mapper);
        return new TupleType(types);
    }
}