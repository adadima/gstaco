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

    struct FuncDecl;
    struct BuiltinFuncDecl;
    struct Definition;
    struct Expression;
    struct TensorVar;
    struct Allocate;
    struct MemAssignment;
    struct Initialize;

    struct ModuleComponent : IR {
        bool is_decl() const;
        std::shared_ptr<FuncDecl> as_decl();

        bool is_builtin() const;
        std::shared_ptr<BuiltinFuncDecl> as_builtin();

        bool is_var() const;
        std::shared_ptr<TensorVar> as_var();

        bool is_def() const;
        std::shared_ptr<Definition> as_def();

        bool is_expr() const;
        std::shared_ptr<Expression> as_expr();

        bool is_allocate() const;
        std::shared_ptr<Allocate> as_allocate();

        bool is_mem_assign() const;
        std::shared_ptr<MemAssignment> as_mem_assign();

        bool is_init() const;
        std::shared_ptr<Initialize> as_init();
    };


    struct IndexVar : Acceptor<IndexVar> {
        std::string name;
        int* coordinate;
        std::shared_ptr<Expression> dimension;

        IndexVar(std::string name, std::shared_ptr<Expression> dimension) :
            name(std::move(name)), dimension(std::move(dimension)), coordinate(nullptr) {}

        IndexVar(std::string name, std::shared_ptr<Expression> dimension, int* coordinate) :
            name(std::move(name)), dimension(std::move(dimension)), coordinate(coordinate) {}

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

    struct TensorVar : Acceptor<TensorVar, ModuleComponent> {
        std::string name;
        std::shared_ptr<TensorType>  type;
        bool is_global;

        explicit TensorVar(std::string name, bool is_global) : name(std::move(name)), type(Type::make<TensorType>()), is_global(is_global) {}
        TensorVar(std::string name, std::shared_ptr<TensorType>  type, bool is_global) : name(std::move(name)), type(std::move(type)), is_global(is_global) {}

        std::string dump() const override;

        std::shared_ptr<TensorType> getType() const;

        bool is_global_var() const;

        template <typename V>
        static std::shared_ptr<einsum::TensorVar> make(std::string name, std::initializer_list<std::shared_ptr<einsum::Expression>> dimensions) {

            auto tType = einsum::IR::make<einsum::TensorType>(
                    einsum::Datatype::make_datatype<V>(),
                    dimensions
            );
            return einsum::IR::make<einsum::TensorVar>(name, tType);
        }

        int getOrder() const {
            return getType()->getOrder();
        }

        std::vector<std::shared_ptr<Expression>> getDimensions() {
            return getType()->getDimensions();
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

        explicit Access(std::string tensor, bool is_global) : tensor(make<TensorVar>(std::move(tensor), std::shared_ptr<TensorType>(new TensorType()), is_global)), indices(std::vector<std::shared_ptr<IndexVar>>()) {}

        Access(std::shared_ptr<TensorVar> tensor, std::vector<std::shared_ptr<IndexVar>> indices) : tensor(std::move(tensor)), indices(std::move(indices)) {}

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const;


        std::string dump() const override;
    };

    struct ReadAccess : Acceptor<ReadAccess, Expression> {
        std::shared_ptr<TensorVar> tensor;
        std::vector<std::shared_ptr<Expression>> indices;

        explicit ReadAccess(std::string tensor, bool is_global) : Base(1), tensor(make<TensorVar>(std::move(tensor), std::shared_ptr<TensorType>(new TensorType()), is_global)), indices(std::vector<std::shared_ptr<Expression>>()) {}

        ReadAccess(std::string tensor, std::vector<std::shared_ptr<Expression>> indices, bool is_global) : Base(1), tensor(make<TensorVar>(std::move(tensor), std::shared_ptr<TensorType>(new TensorType()), is_global)), indices(std::move(indices)) {}

        ReadAccess(std::shared_ptr<TensorVar> tensor, std::vector<std::shared_ptr<Expression>> indices) : Base(1), tensor(std::move(tensor)), indices(std::move(indices)) {}

        std::string dump() const override;

        std::vector<std::shared_ptr<IndexVar>> getIndices() override;

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const override;

        std::shared_ptr<Type> getType() const override;

    };

    struct BuiltinFuncDecl;

    struct Reduction : Acceptor<Reduction> {
        std::shared_ptr<IndexVar> reductionVar;
        std::shared_ptr<BuiltinFuncDecl> reductionOp;
        std::shared_ptr<Expression> reductionInit;

        Reduction(std::shared_ptr<IndexVar> reductionVar, std::shared_ptr<BuiltinFuncDecl> reductionOp, std::shared_ptr<Expression> reductionInit) :
            reductionVar(reductionVar), reductionOp(reductionOp), reductionInit(reductionInit) {
        }

        std::string dump() const override;
    };

    struct Statement : ModuleComponent {

    };

    struct Allocate : Acceptor<Allocate, Statement> {
        std::shared_ptr<TensorVar> tensor;

        explicit Allocate(const std::shared_ptr<TensorVar>&  tensor) :
            tensor(tensor){}

        std::string dump() const override;

    };

    struct Initialize : Acceptor<Initialize, Statement> {
        std::shared_ptr<TensorVar> tensor;

        explicit Initialize(const std::shared_ptr<TensorVar>&  tensor) :
                tensor(tensor){}

        std::string dump() const override;

    };

    struct MemAssignment : Acceptor<MemAssignment, Statement> {
        std::shared_ptr<TensorVar> rhs;
        std::shared_ptr<TensorVar> lhs;

        MemAssignment(std::shared_ptr<TensorVar> rhs, std::shared_ptr<TensorVar> lhs) : rhs(rhs), lhs(lhs) {}

        std::string dump() const override;
    };

    struct Definition : Acceptor<Definition, Statement> {
        std::vector<std::shared_ptr<Reduction>> reduction_list;
        std::vector<std::shared_ptr<Access>> lhs;
        std::shared_ptr<Expression> rhs;

        Definition(std::shared_ptr<Access> lhs, std::shared_ptr<Expression> rhs) :
                    Definition({lhs}, rhs, {}) {}

        Definition(const std::vector<std::shared_ptr<Access>>& lhs,
                   const std::shared_ptr<Expression>& rhs,
                   const std::vector<std::shared_ptr<Reduction>>&  reds) :
                   lhs(lhs),
                   rhs(rhs),
                   reduction_list(reds) {}

        std::string dump() const override;

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const;
        std::set<std::string> getLeftIndexVars() const;
        std::set<std::string> getReductionVars() const;
    };

    struct FuncDecl : Acceptor<FuncDecl, ModuleComponent> {
        std::string funcName;
        std::vector<std::shared_ptr<TensorVar>> inputs;
        std::vector<std::shared_ptr<TensorVar>> outputs;
        std::vector<std::shared_ptr<Statement>> body;

        FuncDecl(std::string funcName, std::vector<std::shared_ptr<TensorVar>> inputs, std::vector<std::shared_ptr<TensorVar>> outputs, std::vector<std::shared_ptr<Statement>> body)
            : funcName(std::move(funcName)), inputs(std::move(inputs)), outputs(std::move(outputs)), body(std::move(body)) {}

        std::string dump() const override;

        std::vector<std::shared_ptr<Type>> getInputType() const;

        std::shared_ptr<TupleType> getOutputType() const;
    };

    struct BuiltinFuncDecl : Acceptor<BuiltinFuncDecl, FuncDecl> {
        std::shared_ptr<Operator> op;

        BuiltinFuncDecl(std::shared_ptr<Operator> op, std::string funcName, std::vector<std::shared_ptr<TensorVar>> inputs, std::vector<std::shared_ptr<TensorVar>> outputs, std::vector<std::shared_ptr<Statement>> body) :
            Base(std::move(funcName), inputs, outputs, body), op(op) {}
    };

    struct AddOperator : Acceptor<AddOperator, BuiltinFuncDecl> {
        AddOperator() : Base(add, "gstaco_add", {}, {}, {}) {}
    };

    struct MulOperator : Acceptor<MulOperator, BuiltinFuncDecl> {
        MulOperator() : Base(mul, "gstaco_mul", {}, {}, {}) {}
    };

    struct AndOperator : Acceptor<AndOperator, BuiltinFuncDecl> {
        AndOperator() : Base(and_, "gstaco_and", {}, {}, {}) {}
    };

    struct OrOperator : Acceptor<OrOperator, BuiltinFuncDecl> {
        OrOperator() : Base(or_, "gstaco_or", {}, {}, {}) {}
    };

    struct MinOperator : Acceptor<MinOperator, BuiltinFuncDecl> {
        MinOperator() : Base(min, "gstaco_min", {}, {}, {}) {}
    };

    struct ChooseOperator : Acceptor<ChooseOperator, BuiltinFuncDecl>{
        ChooseOperator() : Base(choose, "gstaco_choose", {}, {}, {}) {}
    };

    struct Call : Acceptor<Call, Expression> {
        Call(std::string function, std::vector<std::shared_ptr<Expression>> arguments) : Base(1) {
            this->function = IR::make<FuncDecl>(function, std::vector<std::shared_ptr<TensorVar>>(), std::vector<std::shared_ptr<TensorVar>>(), std::vector<std::shared_ptr<Statement>>()),
            this->arguments = std::move(arguments);
        };

        Call(std::shared_ptr<FuncDecl> function, std::vector<std::shared_ptr<Expression>> arguments) : Base(1), function(function), arguments(arguments) {};

        std::string dump() const override;

        std::string dump_args() const;

        std::vector<std::shared_ptr<IndexVar>> getIndices() override;

        std::map<std::string, std::set<std::shared_ptr<Expression>>> getIndexVarDims(IRContext* context) const override;

        std::shared_ptr<Type> getType() const override;

        std::shared_ptr<FuncDecl> function;
        std::vector<std::shared_ptr<Expression>> arguments;
    };

    struct CallStarRepeat : Acceptor<CallStarRepeat, Call> {
        CallStarRepeat(int numIterations, std::string name, std::vector<std::shared_ptr<Expression>> arguments) :
        Base(name, arguments), numIterations(numIterations) {}

        CallStarRepeat(int numIterations, std::shared_ptr<FuncDecl> function, std::vector<std::shared_ptr<Expression>> arguments) :
            Base(function, arguments), numIterations(numIterations) {}

        std::string dump() const override;

        std::string dump_call() const {
            return Call::dump();
        }
        int numIterations;
    };

    struct CallStarCondition : Acceptor<CallStarCondition, Call> {
        CallStarCondition(std::shared_ptr<Expression> stopCondition, std::string name, std::vector<std::shared_ptr<Expression>> arguments) :
                Base(name, arguments), stopCondition(stopCondition) {}

        CallStarCondition(std::shared_ptr<Expression> stopCondition, std::shared_ptr<FuncDecl> function, std::vector<std::shared_ptr<Expression>> arguments) :
                Base(function, arguments), stopCondition(stopCondition) {}

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

        std::vector<std::shared_ptr<TensorVar>> get_globals() const;
    };

    struct IRVisitor {
        virtual void visit(std::shared_ptr<IndexVar> node) = 0;
        virtual void visit(std::shared_ptr<Literal> node) = 0;
        virtual void visit(std::shared_ptr<TensorVar> node) = 0;
        virtual void visit(std::shared_ptr<IndexVarExpr> node) = 0;
        virtual void visit(std::shared_ptr<Access> node) = 0;
        virtual void visit(std::shared_ptr<ReadAccess> node) = 0;
        virtual void visit(std::shared_ptr<BinaryOp> node) = 0;
        virtual void visit(std::shared_ptr<UnaryOp> node) = 0;
        virtual void visit(std::shared_ptr<Definition> node) = 0;
        virtual void visit(std::shared_ptr<Allocate> node) = 0;
        virtual void visit(std::shared_ptr<MemAssignment> node) = 0;
        virtual void visit(std::shared_ptr<Initialize> node) = 0;
        virtual void visit(std::shared_ptr<FuncDecl> node) = 0;
        virtual void visit(std::shared_ptr<AndOperator> node) = 0;
        virtual void visit(std::shared_ptr<OrOperator> node) = 0;
        virtual void visit(std::shared_ptr<AddOperator> node) = 0;
        virtual void visit(std::shared_ptr<MulOperator> node) = 0;
        virtual void visit(std::shared_ptr<MinOperator> node) = 0;
        virtual void visit(std::shared_ptr<ChooseOperator> node) = 0;
        virtual void visit(std::shared_ptr<Call> node) = 0;
        virtual void visit(std::shared_ptr<CallStarRepeat> node) = 0;
        virtual void visit(std::shared_ptr<CallStarCondition> node) = 0;
        virtual void visit(std::shared_ptr<Module> node) = 0;
        virtual void visit(std::shared_ptr<Reduction> node) = 0;
        virtual void visit(std::shared_ptr<Datatype> node) = 0;
        virtual void visit(std::shared_ptr<TensorType> node) = 0;
        virtual void visit(std::shared_ptr<TupleType> node) = 0;
        virtual void visit(std::shared_ptr<Operator> node) = 0;
    };

    template<typename T, typename parent, typename... mixins>
    void Acceptor<T, parent, mixins...>::accept(IRVisitor* v) {
        try {
            v->visit(std::static_pointer_cast<T>(this->shared_from_this()));
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }

    }

    inline std::shared_ptr<MulOperator> mul_red = std::make_shared<MulOperator>();
    inline std::shared_ptr<AddOperator> add_red = std::make_shared<AddOperator>();
    inline std::shared_ptr<AndOperator> and_red = std::make_shared<AndOperator>();
    inline std::shared_ptr<OrOperator>  or_red  = std::make_shared<OrOperator>();
    inline std::shared_ptr<MinOperator> min_red = std::make_shared<MinOperator>();
    inline std::shared_ptr<ChooseOperator> choose_red = std::make_shared<ChooseOperator>();

    struct DefaultIRVisitor : IRVisitor {
        void visit(std::shared_ptr<IndexVar> node) override;
        void visit(std::shared_ptr<Literal> node) override;
        void visit(std::shared_ptr<TensorVar> node) override;
        void visit(std::shared_ptr<IndexVarExpr> node) override;
        void visit(std::shared_ptr<Access> node) override;
        void visit(std::shared_ptr<ReadAccess> node) override;
        void visit(std::shared_ptr<BinaryOp> node) override;
        void visit(std::shared_ptr<UnaryOp> node) override;
        void visit(std::shared_ptr<Definition> node) override;
        void visit(std::shared_ptr<Allocate> node) override;
        void visit(std::shared_ptr<MemAssignment> node) override;
        void visit(std::shared_ptr<Initialize> node) override;
        void visit(std::shared_ptr<FuncDecl> node) override;
        void visit(std::shared_ptr<AndOperator> node) override;
        void visit(std::shared_ptr<OrOperator> node) override;
        void visit(std::shared_ptr<AddOperator> node) override;
        void visit(std::shared_ptr<MulOperator> node) override;
        void visit(std::shared_ptr<MinOperator> node) override;
        void visit(std::shared_ptr<ChooseOperator> node) override;
        void visit(std::shared_ptr<Call> node) override;
        void visit(std::shared_ptr<CallStarRepeat> node) override;
        void visit(std::shared_ptr<CallStarCondition> node) override;
        void visit(std::shared_ptr<Module> node) override;
        void visit(std::shared_ptr<Reduction> node) override;
        void visit(std::shared_ptr<Datatype> node) override;
        void visit(std::shared_ptr<TensorType> node) override;
        void visit(std::shared_ptr<TupleType> node) override;
        void visit(std::shared_ptr<Operator> node) override;

        virtual std::string name();
    };
}

//TODO: implement a switch case
#endif //EINSUM_TACO_IR_H
