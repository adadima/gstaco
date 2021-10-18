//
// Created by Alexandra Dima on 10/16/21.
//

#ifndef EINSUM_TACO_TYPE_H
#define EINSUM_TACO_TYPE_H

#include<vector>

namespace einsum {

    class Type {

    };

    class Datatype : Type {
    public:
        enum Kind {
            Bool,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            UInt128,
            Int8,
            Int16,
            Int32,
            Int64,
            Int128,
            Float32,
            Float64,
            Complex64,
            Complex128
        };

        Datatype(Kind);

        Kind getKind() const;

        bool isUInt() const;

        bool isInt() const;

        bool isFloat() const;

        bool isComplex() const;

        bool isBool() const;

    private:
        Kind kind;
    };

    class Tensortype : Type {
        public:
            int getDimension(int i) const;

            Datatype getElementType() const;

            int numDimensions() const;

        private:
            std::vector<int> dimensions;
            Datatype type;
    };
}
#endif //EINSUM_TACO_TYPE_H
