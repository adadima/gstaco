//
// Created by Alexandra Dima on 26.12.2021.
//

#include "einsum_taco/codegen/codegen_visitor.h"

namespace einsum {
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
        node.tensor->accept(this);
        for (auto &&ind: node.indices) {
            ind->accept(this);
        }
    }
    void CodeGenVisitor::visit(const ReadAccess& node) {
        node.tensor->accept(this);
        for (auto &&ind: node.indices) {
            ind->accept(this);
        }
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

        }
        node.lhs[0]->accept(this);
        oss << " = ";
        visit_reduced_expr(*node.rhs, node.reduction_list);
    }
    void CodeGenVisitor::visit(const FuncDecl& node) {}

    void CodeGenVisitor::visit(const Call& node) {}

    void CodeGenVisitor::visit(const CallStarRepeat& node) {}

    void CodeGenVisitor::visit(const CallStarCondition& node) {}

    void CodeGenVisitor::visit(const Module& node) {}

    void CodeGenVisitor::visit(const Reduction &node) {

    }

    void CodeGenVisitor::visit_reduced_expr(const Expression& expr, const std::vector<std::shared_ptr<Reduction>>& reductions) {
        if (reductions.empty()) {
            expr.accept(this);
            oss << "\n";
        }

        for(int r=0; r < reductions.size(); r++) {
            auto red = reductions[r];
            auto i = red->reductionVar->getName();

            std::string init_var = "init_";
            init_var += i;
            oss << get_indent();
            oss << init_var;
            oss << " = ";
            red->reductionInit->accept(this);
            oss << ";\n";

            oss << get_indent();
            oss << "for(int ";
            oss <<  i;
            oss << "=0; ";
            oss <<  i;
            oss <<  "<";
            red->reductionVar->dimension->accept(this);
            oss << ";"; oss << i; oss << "++) {\n";

            indent();
            if (r == reductions.size() - 1) {
                oss << init_var;
                oss << " = ";
                expr.accept(this);
                oss << "\n";
            }

        }

        for(auto &&red: reductions) {
            unindent();
            oss << get_indent();
            oss << "}\n";
        }
    }

    void CodeGenVisitor::visit(const BinaryOp &node) {
        oss << node.dump();
    }

    void CodeGenVisitor::visit(const UnaryOp &node) {
        oss << node.dump();
    }

//    std::shared_ptr<Definition> reduce_expression(const std::string& init_var, std::shared_ptr<Expression> expr, std::shared_ptr<Operator> op) {
//        std::shared_ptr<Expression> left = IR::make<ReadAccess>(init_var);
//        auto rhs = IR::make<BinaryOp>(left, std::move(expr), std::move(op), );
//        return IR::make<Definition>(IR::make<Access>(init_var), rhs);
//    }
}