//
// Created by Alexandra Dima on 07.01.2022.
//

#ifndef EINSUM_TACO_IR_REWRITER_H
#define EINSUM_TACO_IR_REWRITER_H

#include "einsum_taco/ir/ir.h"
#include "einsum_taco/ir/context.h"

namespace einsum {
    class IRRewriter : public IRMutator {
    protected:
        IRContext* context;

    public:
        std::shared_ptr<Definition> def;
        std::shared_ptr<Expression> expr;
        std::shared_ptr<FuncDecl> func;
        std::shared_ptr<Module> module;
        std::shared_ptr<TensorVar> tensor;
        std::shared_ptr<Access> access;
        std::shared_ptr<IndexVar> index_var;

        explicit IRRewriter(IRContext* context) : context(context) {}

        void visit(IndexVar& node) override;
        void visit(Literal& node) override;
        void visit(ArithmeticExpression& node) override;
        void visit(ModuloExpression& node) override;
        void visit(LogicalExpression& node) override;
        void visit(ComparisonExpression& node) override;
        void visit(NotExpression& node) override;
        void visit(TensorVar& node) override;
        void visit(IndexVarExpr& node) override;
        void visit(Access& node) override;
        void visit(ReadAccess& node) override;
        void visit(Definition& node) override;
        void visit(FuncDecl& node) override;
        void visit(Call& node) override;
        void visit(CallStarRepeat& node) override;
        void visit(CallStarCondition& node) override;
        void visit(Module& node) override;
        void visit(Reduction& node) override;
        std::shared_ptr<ModuleComponent> visit(ModuleComponent& node);
        std::shared_ptr<Expression> visit(Expression& node);

        template<typename T>
        std::shared_ptr<T> rewrite_binary(T& node);

        template<typename T>
        std::shared_ptr<T> rewrite_unary(T& node);
    };

}
#endif //EINSUM_TACO_IR_REWRITER_H
