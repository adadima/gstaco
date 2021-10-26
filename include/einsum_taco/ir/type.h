//
// Created by Alexandra Dima on 10/16/21.
//

#ifndef EINSUM_TACO_TYPE_H
#define EINSUM_TACO_TYPE_H

#include<vector>
#include<map>


namespace einsum {

    struct Type {
        virtual ~Type() = default;
    };

    class Datatype : public Type {
    public:
        enum Kind {
            Bool,
            Int,
            Float,
        };

        explicit Datatype(Kind kind);

        Kind getKind() const;

        bool isInt() const;

        bool isFloat() const;

        bool isBool() const;

        size_t getNumBytes() const;

    private:
        Kind kind;
    };

    struct TupleType : public Type {
        std::vector<Type*> tuple;

        explicit TupleType(std::vector<Type*> tuple) : tuple(std::move(tuple)) {};

        Type* operator[](int pos) {
            return tuple[pos];
        }
    };


    struct TensorType : public Type {
        public:
            int getDimension(int i) const;

            Type* getElementType();

            int getOrder() const;

        private:
            std::vector<int> dimensions;
            Datatype* type;
    };


    enum class OperatorKind {
        Min,
        Max,
        Add,
        Sub,
        Mul,
        Div,
        Or,
        And,
        Choose,
    };

    struct Operator {

        Operator(Datatype* type, OperatorKind op) : type(type), op(op) {}
        Datatype* type;
        OperatorKind op;
    };


}
#endif //EINSUM_TACO_TYPE_H
