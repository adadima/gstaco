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
#include<set>
#include<map>
#include<memory>


namespace einsum {

    class IRContext;
    class IRVisitor;
    class IRMutator;

    struct IR : std::enable_shared_from_this<IR> {

        // TODO: make this a visitor instead
        virtual std::string dump() const = 0;

        virtual void accept(IRVisitor* v) const = 0;

        virtual void accept(IRMutator* v) = 0;

        std::string class_name() const;

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
        void accept(IRVisitor* v) const override;
        void accept(IRMutator* v) override;
        ~Acceptor() override = default;
    };

    struct FuncDecl;
    struct Definition;
    struct Expression;

    struct ModuleComponent : IR {
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

    std::shared_ptr<FuncDecl> as_decl(const std::shared_ptr<ModuleComponent>& component);

    std::shared_ptr<Definition> as_def(const std::shared_ptr<ModuleComponent>& component);

    struct IndexVar : Acceptor<IndexVar> {
        std::string name;
        std::shared_ptr<Expression> dimension;

        IndexVar(std::string name, std::shared_ptr<Expression> dimension) : name(std::move(name)), dimension(std::move(dimension)) {}

        std::string getName() const;

        std::shared_ptr<Expression> getDimension(int i) const;

        std::string dump() const override;
    };

    struct Expression : ModuleComponent {
        int precedence;
        bool isAsymmetric;

        Expression() : precedence(0), isAsymmetric(false) {}

        explicit Expression(int precedence) : precedence(precedence), isAsymmetric(false) {}

        Expression(int precedence, bool isAsymmetric) : precedence(precedence), isAsymmetric(isAsymmetric) {}

        virtual std::shared_ptr<Type> getType() const = 0;
        std::string dump() const override = 0;
        virtual std::vector<std::shared_ptr<IndexVar>> getIndices() = 0;
        virtual std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const = 0;
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

        ~Literal() override {
            free(this->ptr);
        }

        std::string dump() const override;

        std::vector<std::shared_ptr<IndexVar>> getIndices() override;

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const override;

        template<typename T>
        T getValue() const {
             return *(static_cast<T*>(ptr));
        }

        std::shared_ptr<Type> getType() const override;

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
        std::shared_ptr<Type> type;

        BinaryOp(std::shared_ptr<Expression> left_, std::shared_ptr<Expression> right_, std::shared_ptr<Operator> op_, std::shared_ptr<Type> type_) :
                Base(op_->precedence, op_->isAsymmetric), left(std::move(left_)), right(std::move(right_)), op(std::move(op_)) {
            if (!type_) {
                einsum_iassert(op->isArithmetic());
                type = deduce_type();
            } else {
                type = std::move(type_);
            }
        }

        std::string dump() const override;

        std::vector<std::shared_ptr<IndexVar>> getIndices() override;

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const override;

        std::shared_ptr<Type> getType() const override;

    private:
        std::shared_ptr<Type> deduce_type() const {
            if (left->getType()->isFloat() || right->getType()->isFloat()) {
                return std::make_shared<Datatype>(Datatype::Kind::Float);
            }
            return std::make_shared<Datatype>(Datatype::Kind::Int);
        }
    };

//    struct ArithmeticExpression : Acceptor<ArithmeticExpression, BinaryOp> {
//        //template <typename OpT>
//        ArithmeticExpression(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right, std::shared_ptr<Operator> op) :
//                Base(
//                        std::move(left),
//                        std::move(right),
//                        std::move(op),
//                        compute_type(left, right)
//                        ) {
////            static_assert(std::is_base_of<Operator, OpT>::value, "operator of boolean expression has wrong type");
////            static_assert(std::is_base_of<BinaryOperator, OpT>::value, "operator of boolean expression has wrong type");
//        }
//
//        static std::shared_ptr<Type> compute_type(const std::shared_ptr<Expression>& left, const std::shared_ptr<Expression>& right) {
//            if (left->getType()->isFloat() || right->getType()->isFloat()) {
//                return std::make_shared<Datatype>(Datatype::Kind::Float);
//            }
//            return std::make_shared<Datatype>(Datatype::Kind::Int);
//        }
//    };
//
//    struct ModuloExpression : Acceptor<ModuloExpression, BinaryOp> {
//        ModuloExpression(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right, std::shared_ptr<Operator> op) :
//                Base(
//                        std::move(left),
//                        std::move(right),
//                        std::move(op),
//                        Type::make<Datatype>(Datatype::Kind::Int)
//                ) {};
//
//        ModuloExpression(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right) :
//                Base(
//                        std::move(left),
//                        std::move(right),
//                        std::make_shared<ModOp>(),
//                        Type::make<Datatype>(Datatype::Kind::Int)
//                ) {};
//
//        //std::shared_ptr<Type> getType() override;
//    };
//
//    struct LogicalExpression : Acceptor<LogicalExpression, BinaryOp> {
//        //template <typename OpT>
//        LogicalExpression(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right, std::shared_ptr<Operator> op) :
//                Base(
//                        std::move(left),
//                        std::move(right),
//                        std::move(op),
//                        Type::make<Datatype>(Datatype::Kind::Bool)
//                ) {
////                    static_assert(std::is_base_of<Operator, OpT>::value, "operator of boolean expression has wrong type");
////                    static_assert(std::is_base_of<LogicalOperator, OpT>::value, "operator of boolean expression has wrong type");
//        };
//
//        //std::shared_ptr<Type> getType() override;
//    };
//
//    struct ComparisonExpression : Acceptor<ComparisonExpression, BinaryOp> {
//        //template<typename OpT>
//        ComparisonExpression(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right, std::shared_ptr<Operator> op) :
//                Base(
//                        std::move(left),
//                        std::move(right),
//                        std::move(op),
//                        Type::make<Datatype>(Datatype::Kind::Bool)
//                ) {
////                    static_assert(std::is_base_of<Operator, OpT>::value, "operator of arithmetic expression has wrong type");
////                    static_assert(std::is_base_of<ComparisonOperator, OpT>::value, "operator of arithmetic expression has wrong type");
//        };
//
//        //std::shared_ptr<Type> getType() override;
//    };

    struct UnaryOp : Acceptor<UnaryOp, Expression> {
        std::shared_ptr<Expression> expr;
        std::shared_ptr<Operator> op;
        std::shared_ptr<Type> type;

        UnaryOp(std::shared_ptr<Expression> expr, std::shared_ptr<Operator> op, std::shared_ptr<Type> type) : expr(std::move(expr)), op(std::move(op)), type(std::move(type)) {}

        std::string dump() const override;
        std::vector<std::shared_ptr<IndexVar>> getIndices() override;
        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const override;
        std::shared_ptr<Type> getType() const override;
    };

    struct TensorVar : Acceptor<TensorVar> {
        std::string name;
        std::shared_ptr<TensorType>  type;

        explicit TensorVar(std::string name) : name(std::move(name)), type(Type::make<TensorType>()) {}
        TensorVar(std::string name, std::shared_ptr<TensorType>  type) : name(std::move(name)), type(std::move(type)) {}

        std::string dump() const override;

        std::shared_ptr<TensorType> getType() const;

        template <typename V>
        static std::shared_ptr<einsum::TensorVar> make(std::string name, std::initializer_list<std::shared_ptr<einsum::Expression>> dimensions) {

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
        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const override;

        std::string getName() const;

        std::shared_ptr<Expression> getDimension(int i) const;

        std::shared_ptr<Type> getType() const override;
    };

    struct Access: Acceptor<Access> {
        std::shared_ptr<TensorVar> tensor;
        std::vector<std::shared_ptr<IndexVar>> indices;

        explicit Access(std::string tensor) : tensor(make<TensorVar>(std::move(tensor), std::shared_ptr<TensorType>(new TensorType()))), indices(std::vector<std::shared_ptr<IndexVar>>()) {}

        Access(std::shared_ptr<TensorVar> tensor, std::vector<std::shared_ptr<IndexVar>> indices) : tensor(std::move(tensor)), indices(std::move(indices)) {}

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const;


        std::string dump() const override;
    };

    struct ReadAccess : Acceptor<ReadAccess, Expression> {
        std::shared_ptr<TensorVar> tensor;
        std::vector<std::shared_ptr<Expression>> indices;

        explicit ReadAccess(std::string tensor) : Base(1), tensor(make<TensorVar>(std::move(tensor), std::shared_ptr<TensorType>(new TensorType()))), indices(std::vector<std::shared_ptr<Expression>>()) {}

        ReadAccess(std::string tensor, std::vector<std::shared_ptr<Expression>> indices) : Base(1), tensor(make<TensorVar>(std::move(tensor), std::shared_ptr<TensorType>(new TensorType()))), indices(std::move(indices)) {}

        ReadAccess(std::shared_ptr<TensorVar> tensor, std::vector<std::shared_ptr<Expression>> indices) : Base(1), tensor(std::move(tensor)), indices(std::move(indices)) {}

        std::string dump() const override;

        std::vector<std::shared_ptr<IndexVar>> getIndices() override;

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const override;

        std::shared_ptr<Type> getType() const override;

    };

    struct Reduction : Acceptor<Reduction> {
        std::shared_ptr<IndexVar> reductionVar;
        std::shared_ptr<Operator> reductionOp;
        std::shared_ptr<Expression> reductionInit;

        // template<typename OpT>
        Reduction(std::shared_ptr<IndexVar> reductionVar, std::shared_ptr<Operator> reductionOp, std::shared_ptr<Expression> reductionInit) :
            reductionVar(std::move(reductionVar)), reductionOp(std::move(reductionOp)), reductionInit(std::move(reductionInit)) {
            // static_assert(std::is_base_of<Operator, OpT>::value, "operator of reduction has wrong type");
            // static_assert(std::is_base_of<ReductionOperator, OpT>::value, "operator of reduction has wrong type");
        }

        std::string dump() const override;
    };

    struct Definition : Acceptor<Definition, ModuleComponent> {
        Definition(std::shared_ptr<Access> lhs, std::shared_ptr<Expression> rhs) :
                    Definition({std::move(lhs)}, std::move(rhs), {}) {}

//        Definition(std::shared_ptr<Access> lhs, std::shared_ptr<Expression> rhs, const std::vector<std::shared_ptr<Reduction>>& reds) :
//                Definition({std::move(lhs)}, std::move(rhs), reds) {}

        Definition(std::vector<std::shared_ptr<Access>> lhs,
                   std::shared_ptr<Expression> rhs,
                   const std::vector<std::shared_ptr<Reduction>>& reds) :
                   lhs(std::move(lhs)),
                   rhs(std::move(rhs)),
                   reduction_list(reds) {
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

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const;
        std::set<std::string> getLeftIndexVars() const;
        std::set<std::string> getReductionVars() const;
        std::vector<std::shared_ptr<IndexVar>> leftIndices;
        std::vector<std::shared_ptr<IndexVar>> rightIndices;
        std::vector<std::shared_ptr<IndexVar>> reductionVars;
        std::map<std::shared_ptr<IndexVar>, std::shared_ptr<Reduction>> reductions;
        std::vector<std::shared_ptr<Reduction>> reduction_list;
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

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const override;

        std::shared_ptr<Type> getType() const override;

        std::shared_ptr<FuncDecl> function;
        std::vector<std::shared_ptr<Expression>> arguments;
    };

    struct CallStarRepeat : Acceptor<CallStarRepeat, Call> {
        CallStarRepeat(int numIterations, std::shared_ptr<FuncDecl> function, std::vector<std::shared_ptr<Expression>> arguments) :
            Base(std::move(function), std::move(arguments)), numIterations(numIterations) {}

        std::string dump() const override;

        std::string dump_call() const {
            return Call::dump();
        }
        int numIterations;
    };

    struct CallStarCondition : Acceptor<CallStarCondition, Call> {
        CallStarCondition(std::shared_ptr<Expression> stopCondition, std::string name, std::vector<std::shared_ptr<Expression>> arguments) :
                Base(std::move(name), std::move(arguments)), stopCondition(std::move(stopCondition)) {}

        CallStarCondition(std::shared_ptr<Expression> stopCondition, std::shared_ptr<FuncDecl> function, std::vector<std::shared_ptr<Expression>> arguments) :
                Base(std::move(function), std::move(arguments)), stopCondition(std::move(stopCondition)) {}

        std::string dump() const override;

        std::string dump_call() const {
            return Call::dump();
        }

        std::shared_ptr<Expression> stopCondition;
    };

    struct Module : Acceptor<Module> {
        std::vector<std::shared_ptr<ModuleComponent>> decls;

        explicit Module(std::vector<std::shared_ptr<ModuleComponent>> decls) : decls(std::move(decls)) {}

        std::string dump() const override;

        void add(std::shared_ptr<ModuleComponent> decl);
    };

    struct IRVisitor {
        virtual void visit(const IndexVar& node) = 0;
        virtual void visit(const Literal& node) = 0;
        virtual void visit(const TensorVar& node) = 0;
        virtual void visit(const IndexVarExpr& node) = 0;
        virtual void visit(const Access& node) = 0;
        virtual void visit(const ReadAccess& node) = 0;
        virtual void visit(const BinaryOp& node) = 0;
        virtual void visit(const UnaryOp& node) = 0;
        virtual void visit(const Definition& node) = 0;
        virtual void visit(const FuncDecl& node) = 0;
        virtual void visit(const Call& node) = 0;
        virtual void visit(const CallStarRepeat& node) = 0;
        virtual void visit(const CallStarCondition& node) = 0;
        virtual void visit(const Module& node) = 0;
        virtual void visit(const Reduction& node) = 0;
    };

    struct IRMutator {
        virtual void visit(IndexVar& node) = 0;
        virtual void visit(Literal& node) = 0;
        virtual void visit(TensorVar& node) = 0;
        virtual void visit(IndexVarExpr& node) = 0;
        virtual void visit(Access& node) = 0;
        virtual void visit(ReadAccess& node) = 0;
        virtual void visit(BinaryOp& node) = 0;
        virtual void visit(UnaryOp& node) = 0;
        virtual void visit(Definition& node) = 0;
        virtual void visit(FuncDecl& node) = 0;
        virtual void visit(Call& node) = 0;
        virtual void visit(CallStarRepeat& node) = 0;
        virtual void visit(CallStarCondition& node) = 0;
        virtual void visit(Module& node) = 0;
        virtual void visit(Reduction& node) = 0;
    };

    template<typename T, typename parent, typename... mixins>
    void Acceptor<T, parent, mixins...>::accept(IRVisitor* v) const {
        v->visit(static_cast<const T&>(*this));
    }

    template<typename T, typename parent, typename... mixins>
    void Acceptor<T, parent, mixins...>::accept(IRMutator* v) {
        v->visit(static_cast<T&>(*this));
    }
}

//TODO: implement a switch case
#endif //EINSUM_TACO_IR_H
