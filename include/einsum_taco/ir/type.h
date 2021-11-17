//
// Created by Alexandra Dima on 10/16/21.
//

#ifndef EINSUM_TACO_TYPE_H
#define EINSUM_TACO_TYPE_H

#include<vector>
#include<map>
#include<string>
#include<einsum_taco/base/assert.h>


namespace einsum {

    struct Type {
        virtual ~Type() = default;
        virtual bool isInt() const = 0;

        virtual bool isFloat() const = 0;

        virtual bool isBool() const = 0;
        virtual std::string dump() const = 0;

        template<typename T, typename ... Types >
        static std::shared_ptr<T> make(Types... args) {
            return std::make_shared<T>(args...);
        }

        template<typename T, typename ... Types >
        static std::vector<std::shared_ptr<T>> make_vec(Types... args) {
            std::vector<std::shared_ptr<T>> v = {args...};
            return v;
        }
    };

    class Datatype : public Type {
    public:
        enum class Kind {
            Bool,
            Int,
            Float,
        };

        explicit Datatype(Kind kind);

        template <typename T>
        static std::shared_ptr<Datatype> make_datatype() {
            if (std::is_same<T, int>()) {
                return Type::make<Datatype>(Kind::Int);
            }

            if (std::is_same<T, bool>()) {
                return Type::make<Datatype>(Kind::Bool);
            }

            if (std::is_same<T, float>()) {
                return Type::make<Datatype>(Kind::Float);
            }
        }

        Kind getKind() const;

        bool isInt() const override;

        bool isFloat() const override;

        bool isBool() const override;

        std::string dump() const override;

        size_t getNumBytes() const;

    private:
        Kind kind;
    };

    struct TupleType : public Type {
        std::vector<std::shared_ptr<Type>> tuple;

        explicit TupleType(std::vector<std::shared_ptr<Type>> tuple) : tuple(std::move(tuple)) {};

        std::shared_ptr<Type> operator[](int pos) {
            return tuple[pos];
        }

        bool isInt() const override;

        bool isFloat() const override;

        bool isBool() const override;

        std::string dump() const override;
    };


    struct Operator {
        Operator(int precedence, const char* sign) :
                precedence(precedence), sign(sign), reductionSign(""), isAsymmetric(false) {}

        Operator(int precedence, const char* sign, bool isAsymmetric) :
                precedence(precedence), sign(sign), reductionSign(""), isAsymmetric(isAsymmetric) {}

        Operator(int precedence, const char* sign, const char* reductionSign) :
                precedence(precedence), sign(sign), reductionSign(std::move(reductionSign)), isAsymmetric(false) {}

        std::string reductionSign;
        int precedence;
        bool isAsymmetric;
        std::string sign;
    };

    struct BinaryOperator {};

    struct ReductionOperator {};

    struct UnaryOperator {};

    struct LogicalOperator {};

    struct ComparisonOperator {};

    struct AddOp : Operator, BinaryOperator, ReductionOperator {
        AddOp() : Operator(4, "+", "+") {}
    };

    struct SubOp : Operator, BinaryOperator {
        SubOp() : Operator(4, "-", true) {}
    };

    struct MulOp : Operator, BinaryOperator, ReductionOperator {
        MulOp() : Operator(3, "*", "*") {}
    };

    struct DivOp : Operator, BinaryOperator {
        DivOp() : Operator(3, "/", true) {}
    };

    struct ModOp : Operator, BinaryOperator {
        ModOp() : Operator(3, "%") {}
    };

    struct AndOp : Operator, BinaryOperator, LogicalOperator, ReductionOperator {
        AndOp() : Operator(11, "&&", "AND")  {}
    };

    struct OrOp : Operator, BinaryOperator, LogicalOperator, ReductionOperator {
        OrOp() : Operator(12, "||", "OR") {}
    };

    struct NotOp : Operator, LogicalOperator, UnaryOperator {
        NotOp() : Operator(2, "!") {}
    };

    struct LtOp : Operator, ComparisonOperator {
        LtOp() : Operator(6, "<") {}
    };

    struct LteOp : Operator, ComparisonOperator {
        LteOp() : Operator(6, "<=") {}
    };

    struct GtOp : Operator, ComparisonOperator {
        GtOp() : Operator(6, ">") {}
    };

    struct GteOp : Operator, ComparisonOperator {
        GteOp() : Operator(6, ">=") {}
    };

    struct EqOp : Operator, ComparisonOperator {
        EqOp() : Operator(7, "==") {}
    };

    struct NeqOp : Operator, ComparisonOperator {
        NeqOp() : Operator(7, "!=") {}
    };

    struct DimensionType : public Type {
        bool isInt() const override;

        bool isFloat() const override;

        bool isBool() const override;

        std::string dump() const override = 0;
    };


    struct FixedDimension : DimensionType {
        int value;
        explicit FixedDimension(int value) : value(value) {}

        std::string dump() const override;
    };

    struct VariableDimension : DimensionType {
        std::string varName;
        explicit VariableDimension(std::string varName) : varName(std::move(varName)) {}

        std::string dump() const override;
    };

    struct BinaryExpressionDimension : DimensionType {
        std::shared_ptr<DimensionType> left;
        std::shared_ptr<DimensionType> right;
        std::shared_ptr<Operator> op;

        template<typename OpT>
        BinaryExpressionDimension(std::shared_ptr<DimensionType> left,
                                  std::shared_ptr<DimensionType> right,
                                  std::shared_ptr<OpT> op) : left(std::move(left)), right(std::move(right)), op(std::move(op)) {
                    static_assert(std::is_base_of<BinaryOperator, OpT>::value, "operator of binary expression dimension has wrong type");
        }
        std::string dump() const override;
    };

    struct TensorType : public Type {
    public:
        TensorType(std::shared_ptr<Datatype> type, std::vector<std::shared_ptr<einsum::DimensionType>> dimensions) : type(std::move(type)), dimensions(std::move(dimensions)) {}
        std::shared_ptr<einsum::DimensionType> getDimension(int i) const;

        std::shared_ptr<Datatype> getElementType() const;

        int getOrder() const;

        bool isInt() const override;

        bool isFloat() const override;

        bool isBool() const override;

        std::string dump() const override;

    private:
        std::vector<std::shared_ptr<einsum::DimensionType>> dimensions;
        std::shared_ptr<Datatype> type;
    };
}
//TODO: implement a templated utility class which says if something is a reduction or not
// example; is_reduction<MulOp>::value\
//
// struct is_reduction<AddOp> {
//  static const int value = 1;
// }
#endif //EINSUM_TACO_TYPE_H
