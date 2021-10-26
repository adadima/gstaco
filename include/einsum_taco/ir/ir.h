#include <utility>

#include <utility>

//
// Created by Alexandra Dima on 10/13/21.
//

#ifndef EINSUM_TACO_IR_H
#define EINSUM_TACO_IR_H

#include<string>
#include<vector>
#include<einsum_taco/ir/type.h>
#include<einsum_taco/base/assert.h>


namespace einsum {

    class IRVisitor;
    struct IR {};

    template<typename T, typename parent=IR>
    struct IRNode : parent {
        using Base = IRNode<T, parent>;
        using parent :: parent;
        void accept(IRVisitor *v) const;
    };

    struct Expression : IRNode<Expression> {
        explicit Expression(Type* type) : type(type) {}
        Type* type;
    };

    struct Literal : IRNode<Literal, Expression> {

        template<typename T>
        Literal(T value, Datatype* type) {
            einsum_iassert(type->getNumBytes() >= sizeof(T));
            this->type = type;
            void* storage = malloc(type->getNumBytes());
            new (storage) T(value);
            this->ptr = storage;
        }

        template<typename T>
        T* getValue() {
             return static_cast<T*>(ptr);
        }

        Datatype* getDatatype() const {
            return dynamic_cast<Datatype*>(type);
        }

        bool isInt() const;

        bool isFloat() const;

        bool isBool() const;

    private:
        void* ptr;
    };

    struct TensorVar : IRNode<TensorVar> {
        std::string name;
        TensorType*  type;

        TensorType* getType() const;
    };

    struct IndexVar : IRNode<IndexVar> {
        std::string name;
        int dimension;
    };

    struct Access: IRNode<Access> {
        TensorVar tensor;
        std::vector<IndexVar> indices;
    };

    struct ReadAccess : IRNode<ReadAccess, Expression> {
        explicit ReadAccess(Access access) : Base(access.tensor.type->getElementType()), access(std::move(access)) {}
        Access access;
    };

    struct Reduction : IRNode<Reduction> {
        Reduction(IndexVar reductionVar, Operator* reductionOp, Expression* reductionInit) :
            reductionVar(std::move(reductionVar)), reductionOp(reductionOp), reductionInit(reductionInit) {}
        IndexVar reductionVar;
        Operator* reductionOp;
        Expression* reductionInit;
    };

    struct Definition : IRNode<Definition> {
        Definition(Access lhs, Expression* rhs, std::vector<Reduction> reductions) : lhs(std::move(lhs)), rhs(rhs), reductions(std::move(reductions)) {}

        Access lhs;
        Expression* rhs;
        std::vector<Reduction> reductions;
    };

    struct FuncDecl : IRNode<FuncDecl> {
        std::string funcName;
        std::vector<TensorVar> inputs;
        std::vector<TensorVar> outputs;
        std::vector<Definition*> body;

        FuncDecl(std::string funcName, std::vector<TensorVar> inputs, std::vector<TensorVar> outputs, std::vector<Definition*> body)
            : funcName(std::move(funcName)), inputs(std::move(inputs)), outputs(std::move(outputs)), body(std::move(body)) {}

        Type* getInputType() const;

        Type* getOutputType() const;
    };

    struct Call : IRNode<Call, Expression> {
        Call(FuncDecl function, std::vector<Expression*> arguments) : Base(function.getOutputType()), function(std::move(function)), arguments(std::move(arguments)) {};
        FuncDecl function;
        std::vector<Expression*> arguments;
    };


    struct CallStarRepeat : IRNode<CallStarRepeat, Call> {
        CallStarRepeat(int numIterations, FuncDecl function, std::vector<Expression*> arguments) :
            Base(std::move(function), std::move(arguments)), numIterations(numIterations) {}
        int numIterations;
    };

    struct CallStarCondition : IRNode<CallStarCondition, Call> {
        CallStarCondition(Expression* stopCondition, FuncDecl function, std::vector<Expression*> arguments) :
                Base(std::move(function), std::move(arguments)), stopCondition(stopCondition) {}
        Expression* stopCondition;
    };



    struct IRVisitor {
        virtual void accept(const Expression& node) = 0;
        virtual void accept(const FuncDecl& node) = 0;
        virtual void accept(const Definition& node) = 0;
        virtual void accept(const Reduction& node) = 0;
        virtual void accept(const Access& node) = 0;
        virtual void accept(const CallStarRepeat& node) = 0;
        virtual void accept(const CallStarCondition& node) = 0;
        virtual void accept(const Call& node) = 0;
        virtual void accept(const IndexVar& node) = 0;
        virtual void accept(const TensorVar& node) = 0;
    };

    template<typename T, typename parent>
    void IRNode<T, parent>::accept(IRVisitor *v) const {
        v->accept(static_cast<const T&>(*this));
    }
}


#endif //EINSUM_TACO_IR_H
