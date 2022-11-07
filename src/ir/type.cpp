//
// Created by Alexandra Dima on 10/16/21.
//

#include <einsum_taco/ir/type.h>
#include <einsum_taco/ir/ir.h>
#include <algorithm>
#include <iostream>

namespace einsum {
    Datatype::Datatype(Kind kind) : kind(kind) {
    }

    Datatype::Datatype(std::string type_name) {
        einsum_iassert( type_name == "int" || type_name == "float" || type_name == "bool");
        if (type_name == "int") {
            kind = Kind::Int;
        } else if (type_name == "float") {
            kind = Kind::Float;
        } else {
            kind =Kind::Bool;
        }
    }

    Datatype::Kind Datatype::getKind() const {
        return this->kind;
    }

    bool Datatype::isBool() const {
        return this->getKind() == Kind::Bool;
    }

    bool Datatype::isInt() const {
        return this->getKind() == Kind::Int;
    }

    bool Datatype::isFloat() const {
        return this->getKind() == Kind::Float;
    }

    std::string Datatype::dump() const {
        switch(this->getKind()) {
            case Kind::Bool:
                return "bool";
            case Kind::Int:
                return "int";
            case Kind::Float:
                return "float";
        }
    }

//    std::string Datatype::fdump() const {
//        switch(this->getKind()) {
//            case Kind::Bool:
//                return "Bool";
//            case Kind::Int:
//                return "Int64";
//            case Kind::Float:
//                return "Float64";
//        }
//    }

    bool TupleType::isBool() const {
        return false;
    }

    bool TupleType::isInt() const {
        return false;
    }

    bool TupleType::isFloat() const {
        return false;
    }

    std::string TupleType::dump() const {
        std::string types;
        for (int i=0; i < this->tuple.size(); i++) {
            types += this->tuple[i]->dump();
            if (i < this->tuple.size() - 1) {
                types += ", ";
            }
        }
        return "<" + types + ">";
    }

    bool TensorType::isBool() const {
        return false;
    }

    bool TensorType::isInt() const {
        return false;
    }

    bool TensorType::isFloat() const {
        return false;
    }

    std::string TensorType::dump() const {
        std::string dims;
        for (const auto &dimension : this->dimensions) {
            dims += "[" + dimension->dump() + "]";
        }
        return this->getElementType()->dump() + dims;
    }

    size_t Datatype::getNumBytes() const {
        switch(this->getKind()) {
            case Kind::Bool:
                return sizeof(bool);
            case Kind::Int:
                return sizeof(int);
            case Kind::Float:
                return sizeof(double);
        }
    }

    int Datatype::getIntDefault() {
        return 0;
    }

    bool Datatype::getBoolDefault() {
        return false;
    }

    float Datatype::getFloatDefault() {
        return 0;
    }

    std::string Datatype::dumpDefault() const {
        if (isBool()) {
            return std::to_string(getBoolDefault());
        }
        if (isInt()) {
            return std::to_string(getIntDefault());
        }
        if (isFloat()) {
            return std::to_string(getFloatDefault());
        }
        return "";
    }

    std::vector<std::shared_ptr<einsum::Expression>> TensorType::getDimensions() const {
        return this->dimensions;
    }

    std::shared_ptr<einsum::Expression> TensorType::getDimension(int i) const {
        return this->dimensions[i];
    }

    std::shared_ptr<Datatype> TensorType::getElementType() const {
        return this->type;
    }

    int TensorType::getOrder() const {
        return (int) this->dimensions.size();
    }

    constexpr std::array<const char*, 4> arith_ops {"+", "-", "*", "/"};

    bool Operator::isArithmetic() const {
        return std::find(begin(arith_ops), end(arith_ops), sign) != end(arith_ops);
    }

    std::string Operator::get_builtin_name() const {
        return "gstaco_" + class_name();
    }
}
