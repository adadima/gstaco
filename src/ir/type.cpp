//
// Created by Alexandra Dima on 10/16/21.
//

#include <einsum_taco/ir/type.h>

namespace einsum {
    Datatype::Datatype(Kind kind) : kind(kind) {
    }

    Datatype::Kind Datatype::getKind() const {
        return this->kind;
    }

    bool Datatype::isBool() const {
        return this->getKind() == Bool;
    }

    bool Datatype::isInt() const {
        return this->getKind() == Int;
    }

    bool Datatype::isFloat() const {
        return this->getKind() == Float;
    }

    size_t Datatype::getNumBytes() const {
        switch(this->getKind()) {
            case Bool:
                return sizeof(bool);
            case Int:
                return sizeof(int);
            case Float:
                return sizeof(float);
        }
    }

    int TensorType::getDimension(int i) const {
        return this->dimensions[i];
    }

    Type* TensorType::getElementType() {
        return this->type;
    }

    int TensorType::getOrder() const {
        return (int) this->dimensions.size();
    }
}
