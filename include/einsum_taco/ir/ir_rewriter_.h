//
// Created by Alexandra Dima on 20.01.2022.
//

#ifndef EINSUM_TACO_IR_REWRITER__H
#define EINSUM_TACO_IR_REWRITER__H
#include <einsum_taco/ir/ir_rewriter.h>
#include <einsum_taco/ir/ir.h>

namespace einsum {
    struct IndexVarRewriter : public IRRewriter {
        explicit IndexVarRewriter(IRContext* context) : IRRewriter(context) {}

        void visit(IndexVar& node) override;
        void visit(Literal& node) override;
        void visit(TensorVar& node) override;
        void visit(IndexVarExpr& node) override;
        void visit(Access& node) override;
        void visit(ReadAccess& node) override;
        void visit(Definition& node) override;
        void visit(BinaryOp& node) override;
        void visit(UnaryOp& node) override;
        void visit(FuncDecl& node) override;
        void visit(Call& node) override;
        void visit(CallStarRepeat& node) override;
        void visit(CallStarCondition& node) override;
        void visit(Module& node) override;
        void visit(Reduction& node) override;
    protected:
        std::shared_ptr<ModuleComponent> visit(ModuleComponent& node);

    };
}

#endif //EINSUM_TACO_IR_REWRITER__H
