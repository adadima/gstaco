//
// Created by Alexandra Dima on 10/16/21.
//

#ifndef EINSUM_TACO_TYPE_H
#define EINSUM_TACO_TYPE_H

#include <utility>
#include<vector>
#include<map>
#include<string>
#include<array>
#include<einsum_taco/base/assert.h>
#include<memory>


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
        explicit Datatype(std::string type_name);
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

        static std::shared_ptr<Datatype> intType() {
            return make<Datatype>(Kind::Int);
        }

        static std::shared_ptr<Datatype> boolType() {
            return make<Datatype>(Kind::Bool);
        }

        static std::shared_ptr<Datatype> floatType() {
            return make<Datatype>(Kind::Float);
        }

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
        Operator(int precedence, const char* sign, std::shared_ptr<Type> type) :
                precedence(precedence), sign(sign), isAsymmetric(false), type(std::move(type)) {}

        Operator(int precedence, const char* sign, bool isAsymmetric, std::shared_ptr<Type> type) :
                precedence(precedence), sign(sign), isAsymmetric(isAsymmetric), type(std::move(type)) {}

        Operator(int precedence, const char* sign, const char* reductionSign, std::shared_ptr<Type> type) :
                precedence(precedence), sign(sign), reductionSign(reductionSign), isAsymmetric(false), type(std::move(type)) {}

        std::string reductionSign;
        int precedence;
        bool isAsymmetric;
        std::string sign;
        std::shared_ptr<Type> type;

        [[nodiscard]] bool isArithmetic() const;

    };

    struct BinaryOperator {};

    struct UnaryOperator {};

    struct LogicalOperator {};

    struct ComparisonOperator {};

    struct AddOp : Operator, BinaryOperator {
        AddOp() : Operator(4, "+", "+", nullptr) {}
    };

    struct SubOp : Operator, BinaryOperator {
        SubOp() : Operator(4, "-", true, nullptr) {}
    };

    struct MulOp : Operator, BinaryOperator {
        MulOp() : Operator(3, "*", "*", nullptr) {}
    };

    struct DivOp : Operator, BinaryOperator {
        DivOp() : Operator(3, "/", true, nullptr) {}
    };

    struct ModOp : Operator, BinaryOperator {
        ModOp() : Operator(3, "%", Datatype::intType()) {}
    };

    struct AndOp : Operator, BinaryOperator, LogicalOperator {
        AndOp() : Operator(11, "&&", "AND", Datatype::boolType())  {}
    };

    struct OrOp : Operator, BinaryOperator, LogicalOperator {
        OrOp() : Operator(12, "||", "OR", Datatype::boolType()) {}
    };

    struct NotOp : Operator, LogicalOperator, UnaryOperator {
        NotOp() : Operator(2, "!", Datatype::boolType()) {}
    };

    struct LtOp : Operator, ComparisonOperator {
        LtOp() : Operator(6, "<", Datatype::boolType()) {}
    };

    struct LteOp : Operator, ComparisonOperator {
        LteOp() : Operator(6, "<=", Datatype::boolType()) {}
    };

    struct GtOp : Operator, ComparisonOperator {
        GtOp() : Operator(6, ">", Datatype::boolType()) {}
    };

    struct GteOp : Operator, ComparisonOperator {
        GteOp() : Operator(6, ">=", Datatype::boolType()) {}
    };

    struct EqOp : Operator, ComparisonOperator {
        EqOp() : Operator(7, "==", Datatype::boolType()) {}
    };

    struct NeqOp : Operator, ComparisonOperator {
        NeqOp() : Operator(7, "!=", Datatype::boolType()) {}
    };

    struct Expression;


    // TODO: allow users to write their own operators!!
    struct TensorType : public Type {
    public:
        TensorType() : type(Type::make<Datatype>(Datatype::Kind::Int)), dimensions(std::vector<std::shared_ptr<einsum::Expression>>()) {}
        TensorType(std::shared_ptr<Datatype> type, std::vector<std::shared_ptr<einsum::Expression>> dimensions) : type(std::move(type)), dimensions(std::move(dimensions)) {}

        std::vector<std::shared_ptr<einsum::Expression>> getDimensions() const;

        std::shared_ptr<einsum::Expression> getDimension(int i) const;

        std::shared_ptr<Datatype> getElementType() const;

        int getOrder() const;

        bool isInt() const override;

        bool isFloat() const override;

        bool isBool() const override;

        std::string dump() const override;

    private:
        std::vector<std::shared_ptr<einsum::Expression>> dimensions;
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
