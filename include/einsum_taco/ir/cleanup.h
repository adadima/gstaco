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

        void visit(std::shared_ptr<TensorVar> node) override;
        void visit(std::shared_ptr<ReadAccess> node) override;
    };

    struct FuncDeclRewriter : public IRRewriter {
        explicit FuncDeclRewriter(IRContext* context) : IRRewriter(context) {}

        void visit(std::shared_ptr<FuncDecl> node) override;
    };

    struct IndexDimensionRewriter : public IRRewriter {

        explicit IndexDimensionRewriter(IRContext* context) : IRRewriter(context) {}

        void visit(std::shared_ptr<IndexVar> node) override;
        void visit(std::shared_ptr<IndexVarExpr> node) override;
    };

    struct AllocateInserter : public IRRewriter {
        int num_allocations_;

        explicit AllocateInserter(IRContext* context) : IRRewriter(context) {}

        void visit_decl(const std::shared_ptr<FuncDecl>& node) override;
        void visit(std::shared_ptr<Module> node) override;

        int& num_allocations() {
            return num_allocations_;
        }

        void add_allocations() {
            num_allocations() += 1;
        }

        std::vector<std::shared_ptr<Statement>> alloc_and_inst(const std::shared_ptr<TensorVar>& tensor);
    };

    std::shared_ptr<Module> apply_custom_rewriters(std::shared_ptr<Module> mod, const std::vector<IRRewriter*>& rewriters) {
        for (auto& rewriter: rewriters) {
            mod->accept(rewriter);
            mod = std::dynamic_pointer_cast<Module>(rewriter->node_);
        }
        return mod;
    }

    std::shared_ptr<Module> apply_default_rewriters(std::shared_ptr<Module> mod) {
        std::vector<IRRewriter*> rewriters = {
                new TensorVarRewriter(new IRContext()),
                new FuncDeclRewriter(new IRContext()),
                new IndexDimensionRewriter(new IRContext()),
                new AllocateInserter(new IRContext())
        };
        return apply_custom_rewriters(mod, rewriters);
    }


}


#endif //EINSUM_TACO_CLEANUP_H
