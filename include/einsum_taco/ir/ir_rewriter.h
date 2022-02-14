//
// Created by Alexandra Dima on 07.01.2022.
//

#ifndef EINSUM_TACO_IR_REWRITER_H
#define EINSUM_TACO_IR_REWRITER_H

#include "einsum_taco/ir/ir.h"
#include "einsum_taco/ir/context.h"
#include <type_traits>

//TODO: make a rewriter base clarr and overwrite rewrite for different use cases ( see taco, builtit)

namespace einsum {
    class IRRewriter : public IRVisitor {
    protected:
        IRContext* context;

        std::shared_ptr<BinaryOp> rewrite_binary(std::shared_ptr<BinaryOp> node);
        std::shared_ptr<UnaryOp> rewrite_unary(std::shared_ptr<UnaryOp> node);
        std::shared_ptr<ModuleComponent> visit(const std::shared_ptr<ModuleComponent>& node);
        std::shared_ptr<Statement> visit(const std::shared_ptr<Statement>& node);
        virtual void visit_decl(const std::shared_ptr<FuncDecl>& node);

        template<typename T>
        void visit_call(T& node);

    public:
          std::shared_ptr<IR> node_;

        explicit IRRewriter(IRContext* context) : context(context) {}

        template <typename T = IR>
        std::shared_ptr<T> rewrite(const std::shared_ptr<T>& node) {
            if (!node) {
                return node;
            }
            auto tmp = node_;
            node_.reset();
            node->accept(this);
            auto ret = std::dynamic_pointer_cast<T>(node_);
            node_ = tmp;
            return ret;
        }

        void visit(std::shared_ptr<IndexVar> node) override;
        void visit(std::shared_ptr<Literal> node) override;
        void visit(std::shared_ptr<TensorVar> node) override;
        void visit(std::shared_ptr<IndexVarExpr> node) override;
        void visit(std::shared_ptr<Access> node) override;
        void visit(std::shared_ptr<ReadAccess> node) override;
        void visit(std::shared_ptr<Definition> node) override;
        void visit(std::shared_ptr<Allocate> node) override;
        void visit(std::shared_ptr<Instantiation> node) override;
        void visit(std::shared_ptr<BinaryOp> node) override;
        void visit(std::shared_ptr<UnaryOp> node) override;
        void visit(std::shared_ptr<FuncDecl> node) override;
        void visit(std::shared_ptr<Call> node) override;
        void visit(std::shared_ptr<CallStarRepeat> node) override;
        void visit(std::shared_ptr<CallStarCondition> node) override;
        void visit(std::shared_ptr<Module> node) override;
        void visit(std::shared_ptr<Reduction> node) override;
        void visit(std::shared_ptr<Datatype> node) override;
        void visit(std::shared_ptr<TensorType> node) override;
        void visit(std::shared_ptr<TupleType> node) override;
        void visit(std::shared_ptr<Operator> node) override;
    };

}
#endif //EINSUM_TACO_IR_REWRITER_H
