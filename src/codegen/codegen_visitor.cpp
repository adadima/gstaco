//
// Created by Alexandra Dima on 26.12.2021.
//

#include "einsum_taco/codegen/codegen_visitor.h"

namespace einsum {
    template<typename T>
    std::shared_ptr<T> shared_from_ref(const T& ref) {
        return std::dynamic_pointer_cast<T>(ref.shared_from_this());
    }

    void CodeGenVisitor::visit(const IndexVar& node) {
        oss << node.dump();
    }
    void CodeGenVisitor::visit(const Literal& node) {
        oss << node.dump();
    }
    void CodeGenVisitor::visit(const TensorVar& node) {
        oss << node.dump();
    }

    void CodeGenVisitor::visit(const IndexVarExpr& node) {
        node.indexVar->accept(this);
    }

    void CodeGenVisitor::visit(const Access& node) {
        oss << node.dump();
    }

    void CodeGenVisitor::visit(const ReadAccess& node) {
        oss << node.dump();
    }

    //
    // TODO: generate asserts that index var dimensions match
    // frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(OR, 0)
    // Assumes IR has been rewritten to break up definitions of multiple inputs
    // Does not support things like A[i], B[i] = 0; aka the rhs has to be a func call to support multiple outputs
    void CodeGenVisitor::visit(const Definition& node) {
        if (node.lhs.size() > 1) {
            return;  // this is a function call, so rhs can be Call or one of its subtypes
        }
        // otherwise, rhs is a non-call expression.
        auto lhs = node.lhs[0];
        for(auto &&acc : lhs->indices) {
            generate_for_loop(acc);
            indent();
        }

        auto init_var = visit_reduced_expr(node.rhs, node.reduction_list);

        oss << get_indent();
        node.lhs[0]->accept(this);
        oss << " = ";
        oss << init_var;
        oss << ";\n";

        for(auto &&acc: lhs->indices) {
            unindent();
            oss << get_indent();
            oss << "}\n";
        }
    }

    void CodeGenVisitor::visit(const FuncDecl& node) {}

    void CodeGenVisitor::visit(const Call& node) {}

    void CodeGenVisitor::visit(const CallStarRepeat& node) {}

    void CodeGenVisitor::visit(const CallStarCondition& node) {}

    void CodeGenVisitor::visit(const Module& node) {}

    void CodeGenVisitor::visit(const Reduction &node) {
        auto i = node.reductionVar->getName();

        std::string init_var = "init_";
        init_var += i;
        oss << get_indent();
        oss << "auto ";
        oss << init_var;
        oss << " = ";
        node.reductionInit->accept(this);
        oss << ";\n";

        generate_for_loop(node.reductionVar);
    }

    std::string CodeGenVisitor::visit_reduced_expr(const std::shared_ptr<Expression>& expr, const std::vector<std::shared_ptr<Reduction>>& reductions) {
        if (reductions.empty()) {
            oss << get_indent();
            oss << "auto init = ";
            expr->accept(this);
            oss << ";\n";
            return "init";
        }

        std::string var;

        for(int r=0; r < reductions.size(); r++) {
            reductions[r]->accept(this);
            indent();

        }

        for(int r=(reductions.size()-1); r >= 0; r--) {
            auto red = reductions[r];
            auto i = red->reductionVar->getName();

            std::string init_var = "init_";
            init_var += i;

            std::shared_ptr<Expression> exp;
            if (r == (reductions.size() - 1)) {
                exp = reduce_expression(init_var, expr, red->reductionOp);
            } else {
                auto next_var = "init_" + reductions[r+1]->reductionVar->getName();
                exp = reduce_expression(init_var, IR::make<ReadAccess>(next_var), red->reductionOp);
            }

            oss << get_indent();
            oss << init_var;
            oss << " = ";
            exp->accept(this);
            oss << ";\n";

            unindent();
            oss << get_indent();
            oss << "}\n";

            if (!r) {
                var = init_var;
            }
        }

        return var;
    }

    void CodeGenVisitor::visit(const BinaryOp &node) {
        oss << node.dump();
    }

    void CodeGenVisitor::visit(const UnaryOp &node) {
        oss << node.dump();
    }

    std::shared_ptr<Expression> CodeGenVisitor::reduce_expression(const std::string& init_var, std::shared_ptr<Expression> expr, const std::shared_ptr<Operator>& op) {
        std::shared_ptr<Expression> left = IR::make<ReadAccess>(init_var);
        return IR::make<BinaryOp>(left, std::move(expr), op, op->type);
    }

    void CodeGenVisitor::generate_for_loop(const std::shared_ptr<IndexVar>& ivar) {
        oss << get_indent();
        oss << "for(int ";
        oss <<  ivar->getName();
        oss << "=0; ";
        oss <<   ivar->getName();
        oss <<  "<";
        ivar->dimension->accept(this);
        oss << "; "; oss << ivar->getName(); oss << "++) {\n";
    };
}