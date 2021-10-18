//
// Created by Alexandra Dima on 10/16/21.
//

#include <einsum_taco/type.h>

einsum::Datatype::Datatype(Kind kind) : kind(kind) {
}

einsum::Datatype::Kind einsum::Datatype::getKind() const {
    return this->kind;
}

bool einsum::Datatype::isBool() const {
    return getKind() == Bool;
}

bool einsum::Datatype::isUInt() const {
    return getKind() == UInt8 || getKind() == UInt16 || getKind() == UInt32 ||
           getKind() == UInt64 || getKind() == UInt128;
}

bool einsum::Datatype::isInt() const {
    return getKind() == Int8 || getKind() == Int16 || getKind() == Int32 ||
           getKind() == Int64 || getKind() == Int128;
}

bool einsum::Datatype::isFloat() const {
    return getKind() == Float32 || getKind() == Float64;
}

bool einsum::Datatype::isComplex() const {
    return getKind() == Complex64 || getKind() == Complex128;
}

int einsum::Tensortype::getDimension(int i) const {
    return dimensions[i];
}

einsum::Datatype einsum::Tensortype::getElementType() const {
        return type;
}

int einsum::Tensortype::numDimensions() const {
        return (int)dimensions.size();
}
