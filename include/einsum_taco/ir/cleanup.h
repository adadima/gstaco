//
// Created by Alexandra Dima on 20.01.2022.
//

#ifndef EINSUM_TACO_CLEANUP_H
#define EINSUM_TACO_CLEANUP_H

#include "einsum_taco//ir/ir_rewriter.h"
#include <stack>

namespace einsum {
    struct TensorVarRewriter : public IRRewriter {
        explicit TensorVarRewriter(IRContext* context) : IRRewriter(context) {}

        void visit(TensorVar& node) override;
        void visit(ReadAccess& node) override;
    };

    struct FuncDeclRewriter : public IRRewriter {
        explicit FuncDeclRewriter(IRContext* context) : IRRewriter(context) {}

        void visit(FuncDecl& node) override;
    };

    struct IndexDimensionRewriter : public IRRewriter {

        explicit IndexDimensionRewriter(IRContext* context) : IRRewriter(context) {}

        void visit(IndexVar& node) override;
        void visit(IndexVarExpr& node) override;
    };

    std::shared_ptr<Module> apply_rewriters(std::shared_ptr<Module> mod, const std::vector<IRRewriter*>& rewriters) {
        for (auto& rewriter: rewriters) {
            mod->accept(rewriter);
            mod = rewriter->module;
        }
        return mod;
    }
}


#endif //EINSUM_TACO_CLEANUP_H
