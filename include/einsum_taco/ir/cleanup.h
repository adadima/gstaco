//
// Created by Alexandra Dima on 20.01.2022.
//

#ifndef EINSUM_TACO_CLEANUP_H
#define EINSUM_TACO_CLEANUP_H

#include "einsum_taco//ir/ir_rewriter.h"
#include <stack>
#include <unordered_set>

namespace einsum {
    struct TensorVarRewriter : public IRRewriter {
        explicit TensorVarRewriter(IRContext* context) : IRRewriter(context) {}

        void visit(std::shared_ptr<TensorVar> node) override;
        void visit(std::shared_ptr<ReadAccess> node) override;
    };

    struct IndexDimensionRewriter : public IRRewriter {

        explicit IndexDimensionRewriter(IRContext* context) : IRRewriter(context) {}

        void visit(std::shared_ptr<IndexVar> node) override;
        void visit(std::shared_ptr<IndexVarExpr> node) override;
    };

    struct AllocateInserter : public IRRewriter {
        int num_allocations_ = 0;

        explicit AllocateInserter(IRContext* context) : IRRewriter(context) {}

        void visit_decl(const std::shared_ptr<FuncDecl>& node) override;
        void visit(std::shared_ptr<Module> node) override;

        int& num_allocations() {
            return num_allocations_;
        }

        void add_allocations() {
            num_allocations() += 1;
        }
    };

    struct ReductionOpGenerator : public IRRewriter {
        std::unordered_set<std::shared_ptr<BuiltinFuncDecl>> reduction_ops;

        explicit ReductionOpGenerator(IRContext* context) : IRRewriter(context) {}

        void visit(std::shared_ptr<Reduction> node) override;
        void visit(std::shared_ptr<Module> node) override;
    };

    struct CallRewriter : public IRRewriter {
        std::map<std::shared_ptr<ModuleComponent>,std::shared_ptr<Call>> inner_calls;
        std::vector<std::shared_ptr<ModuleComponent>> temporaries;
        int call_id = 0;
        int call_output_idx = 0;
        explicit CallRewriter(IRContext* context) : IRRewriter(context) {}

        void visit_decl(const std::shared_ptr<FuncDecl>& node) override;
        void visit_call(std::shared_ptr<Call> node) override;
    };

    struct CallStarConditionProcessor : public IRRewriter {
        std::set<std::shared_ptr<TensorVar>> condition_tensors;
        std::vector<std::vector<std::shared_ptr<Reduction>>> reductions;
        std::set<std::string> index_vars;
        int call_id = 0;
        bool inside_stop_condition = false;

        explicit CallStarConditionProcessor(IRContext* context) : IRRewriter(context) {}

        void visit_decl(const std::shared_ptr<FuncDecl>& node) override;
        void visit(std::shared_ptr<CallStarCondition> node) override;
        void visit_call(std::shared_ptr<Call> node) override;
        void visit(std::shared_ptr<ReadAccess> node) override;
    };

    struct DefinitionSplitter : public IRRewriter {
        std::set<std::shared_ptr<Definition>> new_defs;
        int def_id = 0;

        explicit DefinitionSplitter(IRContext* context) : IRRewriter(context) {}

        void visit(std::shared_ptr<Definition> node) override;
        void visit(std::shared_ptr<MultipleOutputDefinition> node) override;
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
                new IRRewriter(new IRContext()),
                new TensorVarRewriter(new IRContext()),
                new IndexDimensionRewriter(new IRContext()),
                new AllocateInserter(new IRContext()),
                new CallRewriter(new IRContext()),
                new CallStarConditionProcessor(new IRContext()),
                new DefinitionSplitter(new IRContext()),
                new ReductionOpGenerator(new IRContext())
        };
        return apply_custom_rewriters(mod, rewriters);
    }


}


#endif //EINSUM_TACO_CLEANUP_H
