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
    struct IR : std::enable_shared_from_this<IR> {
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

        virtual ~IR() = default;
    };

    template<typename T, typename parent = IR, typename... mixins>
    struct Acceptor : parent, mixins... {
        using Base = Acceptor<T, parent, mixins...>;
        using parent :: parent;
        void accept(std::shared_ptr<IRVisitor> v) const;
        virtual ~Acceptor() = default;
    };

    struct FuncDecl;
    struct Definition;
    struct Expression;

    struct ModuleComponent : Acceptor<ModuleComponent> {
        bool is_decl() const;
        FuncDecl& as_decl();
        const FuncDecl& as_decl() const;

        bool is_def() const;
        Definition& as_def();
        const Definition& as_def() const;

        bool is_expr() const;
        Expression& as_expr();
        const Expression& as_expr() const;
    };

    struct IndexVar : Acceptor<IndexVar> {
        std::string name;
        int dimension;

        IndexVar(std::string name, int dimension) : name(std::move(name)), dimension(dimension) {}

        std::string dump() const override;
    };

    struct Expression : Acceptor<Expression, ModuleComponent> {
        int precedence;
        bool isAsymmetric;

        Expression() : precedence(0), isAsymmetric(false) {}

        explicit Expression(int precedence) : precedence(precedence), isAsymmetric(false) {}

        Expression(int precedence, bool isAsymmetric) : precedence(precedence), isAsymmetric(isAsymmetric) {}

        virtual std::shared_ptr<Type> getType() = 0;
        std::string dump() const override = 0;
        virtual std::vector<std::shared_ptr<IndexVar>> getIndices() = 0;
    };

    struct Literal : Acceptor<Literal, Expression> {

        template<typename T>
        Literal(T value, std::shared_ptr<Datatype> type) : Base(0), type(std::move(type)) {
            size_t numBytes = this->getDatatype()->getNumBytes();
            einsum_iassert( numBytes >= sizeof(T));
            void* storage = malloc(numBytes);
            new (storage) T(value);
            this->ptr = storage;
        }

        ~Literal() {
            free(this->ptr);
        }

        std::string dump() const override;

        std::vector<std::shared_ptr<IndexVar>> getIndices() override;

        template<typename T>
        T getValue() const {
             return *(static_cast<T*>(ptr));
        }

        std::shared_ptr<Type> getType() override;

        std::shared_ptr<Datatype> getDatatype() const {
            return std::dynamic_pointer_cast<Datatype>(type);
        }

        bool isInt() const;

        bool isFloat() const;

        bool isBool() const;

    private:
        void* ptr;
        std::shared_ptr<Datatype> type;
    };

    struct BinaryOp : Acceptor<BinaryOp, Expression> {
        std::shared_ptr<Expression> left;
        std::shared_ptr<Expression> right;
        std::shared_ptr<Operator> op;

        BinaryOp(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right, std::shared_ptr<Operator> op) : Base(op->precedence, op->isAsymmetric), left(std::move(left)), right(std::move(right)), op(std::move(op)) {}

        std::string dump() const override;

        std::vector<std::shared_ptr<IndexVar>> getIndices() override;

        std::shared_ptr<Type> getType() override = 0;
    };

    struct ArithmeticExpression : Acceptor<ArithmeticExpression, BinaryOp> {
        template <typename OpT>
        ArithmeticExpression(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right, std::shared_ptr<OpT> op) :
                Base(
                        std::move(left),
                        std::move(right),
                        std::move(op)
                        ) {
            static_assert(std::is_base_of<Operator, OpT>::value, "operator of boolean expression has wrong type");
            static_assert(std::is_base_of<BinaryOperator, OpT>::value, "operator of boolean expression has wrong type");
        }
        std::shared_ptr<Type> getType() override;
    };

    struct ModuloExpression : Acceptor<ModuloExpression, BinaryOp> {
        ModuloExpression(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right) :
                Base(
                        std::move(left),
                        std::move(right),
                        std::make_shared<ModOp>()
                ) {};

        std::shared_ptr<Type> getType() override;
    };

    struct LogicalExpression : Acceptor<LogicalExpression, BinaryOp> {
        template <typename OpT>
        LogicalExpression(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right, std::shared_ptr<OpT> op) :
                Base(
                        std::move(left),
                        std::move(right),
                        op
                ) {
                    static_assert(std::is_base_of<Operator, OpT>::value, "operator of boolean expression has wrong type");
                    static_assert(std::is_base_of<LogicalOperator, OpT>::value, "operator of boolean expression has wrong type");
        };

        std::shared_ptr<Type> getType() override;
    };

    struct ComparisonExpression : Acceptor<ComparisonExpression, BinaryOp> {
        template<typename OpT>
        ComparisonExpression(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right, std::shared_ptr<OpT> op) :
                Base(
                        std::move(left),
                        std::move(right),
                        op
                ) {
                    static_assert(std::is_base_of<Operator, OpT>::value, "operator of arithmetic expression has wrong type");
                    static_assert(std::is_base_of<ComparisonOperator, OpT>::value, "operator of arithmetic expression has wrong type");
        };

        std::shared_ptr<Type> getType() override;
    };

    struct UnaryOp : Acceptor<UnaryOp, Expression> {
        std::shared_ptr<Expression> expr;
        std::shared_ptr<Operator> op;

        UnaryOp(std::shared_ptr<Expression> expr, std::shared_ptr<Operator> op) : Base(op->precedence), expr(std::move(expr)), op(std::move(op)) {}

        std::string dump() const override;
        std::vector<std::shared_ptr<IndexVar>> getIndices() override;
    };

    struct NotExpression : Acceptor<NotExpression, UnaryOp> {
        explicit NotExpression(std::shared_ptr<Expression> expr) :
                Base(
                        std::move(expr),
                        std::make_shared<NotOp>()
                ) {};

        std::shared_ptr<Type> getType() override;
    };

    struct TensorVar : Acceptor<TensorVar> {
        std::string name;
        std::shared_ptr<TensorType>  type;

        TensorVar(std::string name, std::shared_ptr<TensorType>  type) : name(std::move(name)), type(std::move(type)) {}

        std::string dump() const override;

        std::shared_ptr<TensorType> getType() const;

        template <typename V>
        static std::shared_ptr<einsum::TensorVar> make(std::string name, std::initializer_list<std::shared_ptr<einsum::DimensionType>> dimensions) {

            auto tType = einsum::IR::make<einsum::TensorType>(
                    einsum::Datatype::make_datatype<V>(),
                    dimensions
            );
            return einsum::IR::make<einsum::TensorVar>(name, tType);
        }
    };


    struct IndexVarExpr : Acceptor<IndexVarExpr, Expression> {
        std::shared_ptr<IndexVar> indexVar;
        explicit IndexVarExpr(std::shared_ptr<IndexVar> indexVar) : Base(0), indexVar(std::move(indexVar)) {};
        std::string dump() const override;
        std::vector<std::shared_ptr<IndexVar>> getIndices() override;
        std::shared_ptr<Type> getType() override;
    };

    struct Access: Acceptor<Access> {
        std::shared_ptr<TensorVar> tensor;
        std::vector<std::shared_ptr<IndexVar>> indices;

        Access(std::shared_ptr<TensorVar> tensor, std::vector<std::shared_ptr<IndexVar>> indices) : tensor(std::move(tensor)), indices(std::move(indices)) {}

        std::string dump() const override;
    };

    struct ReadAccess : Acceptor<ReadAccess, Expression> {
        std::shared_ptr<TensorVar> tensor;
        std::vector<std::shared_ptr<Expression>> indices;

        explicit ReadAccess(std::string tensor) : Base(1), tensor(make<TensorVar>(tensor, std::shared_ptr<TensorType>(new TensorType()))), indices(std::vector<std::shared_ptr<Expression>>()) {}

        ReadAccess(std::string tensor, std::vector<std::shared_ptr<Expression>> indices) : Base(1), tensor(make<TensorVar>(tensor, std::shared_ptr<TensorType>(new TensorType()))), indices(std::move(indices)) {}

        ReadAccess(std::shared_ptr<TensorVar> tensor, std::vector<std::shared_ptr<Expression>> indices) : Base(1), tensor(std::move(tensor)), indices(std::move(indices)) {}

        std::string dump() const override;
        std::vector<std::shared_ptr<IndexVar>> getIndices() override;
        std::shared_ptr<Type> getType() override;

    };

    struct Reduction : Acceptor<Reduction> {
        std::shared_ptr<IndexVar> reductionVar;
        std::shared_ptr<Operator> reductionOp;
        std::shared_ptr<Expression> reductionInit;

        template<typename OpT>
        Reduction(std::shared_ptr<IndexVar> reductionVar, std::shared_ptr<OpT> reductionOp, std::shared_ptr<Expression> reductionInit) :
            reductionVar(std::move(reductionVar)), reductionOp(std::move(reductionOp)), reductionInit(std::move(reductionInit)) {
            static_assert(std::is_base_of<Operator, OpT>::value, "operator of reduction has wrong type");
            static_assert(std::is_base_of<ReductionOperator, OpT>::value, "operator of reduction has wrong type");
        }

        std::string dump() const override;
    };

    struct Definition : Acceptor<Definition, ModuleComponent> {
        Definition(std::vector<std::shared_ptr<Access>> lhs,
                   std::shared_ptr<Expression> rhs,
                   std::vector<std::shared_ptr<Reduction>> reds) :
                   lhs(std::move(lhs)),
                   rhs(std::move(rhs)) {
            for (auto & lh : this->lhs) {
                auto inds = lh->indices;
                for (const auto & ind : inds) {
                    leftIndices.push_back(ind);
                }
            }
            rightIndices = this->rhs->getIndices();

            for (auto &red : reds) {
                reductionVars.push_back(red->reductionVar);
                reductions[red->reductionVar] = red;
            }
        }

        std::string dump() const override;

        std::vector<std::shared_ptr<IndexVar>> leftIndices;
        std::vector<std::shared_ptr<IndexVar>> rightIndices;
        std::vector<std::shared_ptr<IndexVar>> reductionVars;
        std::map<std::shared_ptr<IndexVar>, std::shared_ptr<Reduction>> reductions;
        std::vector<std::shared_ptr<Access>> lhs;
        std::shared_ptr<Expression> rhs;
    };

    struct FuncDecl : Acceptor<FuncDecl, ModuleComponent> {
        std::string funcName;
        std::vector<std::shared_ptr<TensorVar>> inputs;
        std::vector<std::shared_ptr<TensorVar>> outputs;
        std::vector<std::shared_ptr<Definition>> body;

        FuncDecl(std::string funcName, std::vector<std::shared_ptr<TensorVar>> inputs, std::vector<std::shared_ptr<TensorVar>> outputs, std::vector<std::shared_ptr<Definition>> body)
            : funcName(std::move(funcName)), inputs(std::move(inputs)), outputs(std::move(outputs)), body(std::move(body)) {}

        std::string dump() const override;

        std::vector<std::shared_ptr<Type>> getInputType() const;

        std::vector<std::shared_ptr<Type>> getOutputType() const;
    };

    struct Call : Acceptor<Call, Expression> {
        Call(std::string function, std::vector<std::shared_ptr<Expression>> arguments) : Base(1) {
            this->function = IR::make<FuncDecl>(std::move(function), std::vector<std::shared_ptr<TensorVar>>(), std::vector<std::shared_ptr<TensorVar>>(), std::vector<std::shared_ptr<Definition>>()),
            this->arguments = std::move(arguments);
        };

        Call(std::shared_ptr<FuncDecl> function, std::vector<std::shared_ptr<Expression>> arguments) : Base(1), function(std::move(function)), arguments(std::move(arguments)) {};

        std::string dump() const override;

        std::string dump_args() const;

        std::vector<std::shared_ptr<IndexVar>> getIndices() override;

        std::shared_ptr<Type> getType() override;

        std::shared_ptr<FuncDecl> function;
        std::vector<std::shared_ptr<Expression>> arguments;
    };


    struct CallStarRepeat : Acceptor<CallStarRepeat, Call> {
        CallStarRepeat(int numIterations, std::shared_ptr<FuncDecl> function, std::vector<std::shared_ptr<Expression>> arguments) :
            Base(std::move(function), std::move(arguments)), numIterations(numIterations) {}

        std::string dump() const override;

        int numIterations;
    };

    struct CallStarCondition : Acceptor<CallStarCondition, Call> {
        CallStarCondition(std::shared_ptr<Expression> stopCondition, std::string name, std::vector<std::shared_ptr<Expression>> arguments) :
                Base(std::move(name), std::move(arguments)), stopCondition(std::move(stopCondition)) {}

        CallStarCondition(std::shared_ptr<Expression> stopCondition, std::shared_ptr<FuncDecl> function, std::vector<std::shared_ptr<Expression>> arguments) :
                Base(std::move(function), std::move(arguments)), stopCondition(std::move(stopCondition)) {}

        std::string dump() const override;

        std::shared_ptr<Expression> stopCondition;
    };

    struct Module : Acceptor<Module> {
        std::vector<std::shared_ptr<ModuleComponent>> decls;

        explicit Module(std::vector<std::shared_ptr<ModuleComponent>> decls) : decls(std::move(decls)) {}

        std::string dump() const override;

        void add(std::shared_ptr<ModuleComponent> decl);
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

    template<typename T, typename parent, typename... mixins>
    void Acceptor<T, parent, mixins...>::accept(std::shared_ptr<IRVisitor> v) const {
        v->accept(static_cast<const T&>(*this));
    }
}

//TODO: typdef the shared ptr
//TODO: write test cases
//TODO: represent modules and built ins already in scope
//TODO: implement a switch case
#endif //EINSUM_TACO_IR_H
