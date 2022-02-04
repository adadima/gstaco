//
// Created by Alexandra Dima on 10/16/21.
//

#ifndef EINSUM_TACO_TYPE_H
#define EINSUM_TACO_TYPE_H

#include<utility>
#include<vector>
#include<map>
#include<string>
#include<array>
#include<einsum_taco/base/assert.h>
#include<einsum_taco/ir/acceptor.h>
#include<memory>


namespace einsum {

    struct Type : IR {
        virtual void accept(IRVisitor* v) override = 0;

        virtual ~Type() = default;

        virtual bool isInt() const = 0;

        virtual bool isFloat() const = 0;

        virtual bool isBool() const = 0;

        virtual std::string dump() const override = 0;

        template<typename T, typename ... Types>
        static std::shared_ptr<T> make(Types... args)  {
            return std::make_shared<T>(args...);
        }

        template<typename T, typename ... Types>
        static std::vector<std::shared_ptr<T>> make_vec(Types... args)  {
            std::vector<std::shared_ptr<T>> v = {args...};
            return v;
        }
    };

    class Datatype : public Acceptor<Datatype, Type> {
    public:
        enum class Kind {
            Bool,
            Int,
            Float,
        };

        explicit Datatype(Kind kind);

        explicit Datatype(std::string type_name);

        template<typename T>
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

    struct TupleType : Acceptor<TupleType, Type> {
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


    struct Operator : Acceptor<Operator> {
        Operator(int precedence, const char *sign, std::shared_ptr<Type> type) :
                precedence(precedence), sign(sign), isAsymmetric(false), type(std::move(type)) {}

        Operator(int precedence, const char *sign, bool isAsymmetric, std::shared_ptr<Type> type) :
                precedence(precedence), sign(sign), isAsymmetric(isAsymmetric), type(std::move(type)) {}

        Operator(int precedence, const char *sign, const char *reductionSign, std::shared_ptr<Type> type) :
                precedence(precedence), sign(sign), reductionSign(reductionSign), isAsymmetric(false),
                type(std::move(type)) {}

        std::string reductionSign;
        int precedence;
        bool isAsymmetric;
        std::string sign;
        std::shared_ptr<Type> type;

        std::string dump() const {
            return sign;
        }

        [[nodiscard]] bool isArithmetic() const;
    };


    struct AddOp : Operator {
        AddOp() : Operator(4, "+", "+", nullptr) {}
    };

    struct SubOp : Operator {
        SubOp() : Operator(4, "-", true, nullptr) {}
    };

    struct MulOp : Operator {
        MulOp() : Operator(3, "*", "*", nullptr) {}
    };

    struct DivOp : Operator {
        DivOp() : Operator(3, "/", true, nullptr) {}
    };

    struct ModOp : Operator {
        ModOp() : Operator(3, "%", Datatype::intType()) {}
    };

    struct AndOp : Operator {
        AndOp() : Operator(11, "&&", "AND", Datatype::boolType()) {}
    };

    struct OrOp : Operator {
        OrOp() : Operator(12, "||", "OR", Datatype::boolType()) {}
    };

    struct NotOp : Operator {
        NotOp() : Operator(2, "!", Datatype::boolType()) {}
    };

    struct LtOp : Operator {
        LtOp() : Operator(6, "<", Datatype::boolType()) {}
    };

    struct LteOp : Operator {
        LteOp() : Operator(6, "<=", Datatype::boolType()) {}
    };

    struct GtOp : Operator {
        GtOp() : Operator(6, ">", Datatype::boolType()) {}
    };

    struct GteOp : Operator {
        GteOp() : Operator(6, ">=", Datatype::boolType()) {}
    };

    struct EqOp : Operator {
        EqOp() : Operator(7, "==", Datatype::boolType()) {}
    };

    struct NeqOp : Operator {
        NeqOp() : Operator(7, "!=", Datatype::boolType()) {}
    };

    struct Expression;


    // TODO: allow users to write their own operators!!
    struct TensorType : public Acceptor<TensorType, Type> {
    public:
        TensorType() : type(Type::make<Datatype>(Datatype::Kind::Int)),
                       dimensions(std::vector<std::shared_ptr<einsum::Expression>>()) {}

        TensorType(std::shared_ptr<Datatype> type, std::vector<std::shared_ptr<einsum::Expression>> dimensions) : type(
                std::move(type)), dimensions(std::move(dimensions)) {}

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
