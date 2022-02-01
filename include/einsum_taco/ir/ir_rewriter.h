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

        template<typename T>
        std::shared_ptr<T> shared_from_ref(T& ref);
        std::shared_ptr<BinaryOp> rewrite_binary(std::shared_ptr<BinaryOp> node);
        std::shared_ptr<UnaryOp> rewrite_unary(std::shared_ptr<UnaryOp> node);

        template<typename T>
        void visit_call(T& node);

    public:
        std::shared_ptr<Definition> def;
        std::shared_ptr<Expression> expr;
        std::shared_ptr<FuncDecl> func;
        std::shared_ptr<Module> module;
        std::shared_ptr<TensorVar> tensor;
        std::shared_ptr<Access> access;
        std::shared_ptr<IndexVar> index_var;
        std::shared_ptr<Reduction> reduction;

        explicit IRRewriter(IRContext* context) : context(context) {}

        template<typename T>
        std::shared_ptr<T>& get(const std::shared_ptr<T>& node) {
            if constexpr(std::is_same_v<T, IndexVar>) {
                return index_var;
            } else
            if constexpr(std::is_base_of_v<Access, T>) {
                return access;
            } else
            if constexpr(std::is_base_of_v<TensorVar, T>) {
                return tensor;
            } else
            if constexpr(std::is_base_of_v<Module, T>) {
                return module;
            } else
            if constexpr(std::is_base_of_v<FuncDecl, T>) {
                return func;
            } else
            if constexpr(std::is_base_of_v<Definition, T>) {
                return def;
            } else
            if constexpr(std::is_base_of_v<Expression, T>) {
                return expr;
            } else
            if constexpr(std::is_base_of_v<Reduction, T>) {
                return reduction;
            } else {

            }
        }

        template <typename T = IR>
        std::shared_ptr<T> rewrite(const std::shared_ptr<T>& node) {
            if (!node) {
                return node;
            }
            auto &ref = get(node);
            auto tmp = ref;
            ref.reset();
            node->accept(this);
            auto ret = std::dynamic_pointer_cast<T>(ref);
            ref = tmp;
            return ret;
        }

        void visit(std::shared_ptr<IndexVar> node) override;
        void visit(std::shared_ptr<Literal> node) override;
        void visit(std::shared_ptr<TensorVar> node) override;
        void visit(std::shared_ptr<IndexVarExpr> node) override;
        void visit(std::shared_ptr<Access> node) override;
        void visit(std::shared_ptr<ReadAccess> node) override;
        void visit(std::shared_ptr<Definition> node) override;
        void visit(std::shared_ptr<BinaryOp> node) override;
        void visit(std::shared_ptr<UnaryOp> node) override;
        void visit(std::shared_ptr<FuncDecl> node) override;
        void visit(std::shared_ptr<Call> node) override;
        void visit(std::shared_ptr<CallStarRepeat> node) override;
        void visit(std::shared_ptr<CallStarCondition> node) override;
        void visit(std::shared_ptr<Module> node) override;
        void visit(std::shared_ptr<Reduction> node) override;
    };

}
#endif //EINSUM_TACO_IR_REWRITER_H
