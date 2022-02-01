//
// Created by Alexandra Dima on 26.12.2021.
//

#include "einsum_taco/codegen/codegen_visitor.h"

#include<iostream>

namespace einsum {

    void CodeGenVisitor::visit(std::shared_ptr<IndexVar> node) {
        oss << node->dump();
    }
    void CodeGenVisitor::visit(std::shared_ptr<Literal> node) {
        oss << node->dump();
    }
    void CodeGenVisitor::visit(std::shared_ptr<TensorVar> node) {
        oss << node->dump();
    }

    void CodeGenVisitor::visit(std::shared_ptr<IndexVarExpr> node) {
        node->indexVar->accept(this);
    }

    void CodeGenVisitor::visit(std::shared_ptr<Access> node) {
        oss << node->dump();
    }

    void CodeGenVisitor::visit(std::shared_ptr<ReadAccess> node) {
        oss << node->tensor->name;
        for (const auto &indice : node->indices) {
            oss << "[";
            indice->accept(this);
            oss << "]";
        }
    }

    //
    // TODO: generate asserts that index var dimensions match
    // frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(OR, 0)
    // Assumes IR has been rewritten to break up definitions of multiple inputs
    // Does not support things like A[i], B[i] = 0; aka the rhs has to be a func call to support multiple outputs
    void CodeGenVisitor::visit(std::shared_ptr<Definition> node) {
        for(int a=0; a < node->lhs.size(); a++) {
            oss << get_indent();
            oss << "{\n";
            indent();

            auto lhs = node->lhs[a];
            for(auto &&acc : lhs->indices) {
                generate_for_loop(acc->getName(), acc->dimension);
                indent();
            }

            auto init_var = visit_reduced_expr(node->rhs, node->reduction_list);

            oss << get_indent();
            lhs->accept(this);
            oss << " = ";
            if (node->lhs.size() > 1) {
                oss << "std::get<";
                oss << std::to_string(a);
                oss << ">(";
                oss << init_var;
                oss << ")";
            } else {
                oss << init_var;
            }
            oss << ";\n";

            for(auto &&acc: lhs->indices) {
                unindent();
                oss << get_indent();
                oss << "}\n";
            }
            unindent();
            oss << get_indent();
            oss << "}\n";
        }
    }

    void CodeGenVisitor::visit(std::shared_ptr<FuncDecl> node) {}

    void CodeGenVisitor::visit(std::shared_ptr<Call> node) {
        oss << node->function->funcName;
        oss << "(";
        for (int i=0; i < node->arguments.size(); i++) {
            if (i > 0) {
                oss << ", ";
            }
            node->arguments[i]->accept(this);
        }
        oss << ")";
    }

    void CodeGenVisitor::get_lambda_return(std::string output_type, int num_outputs) {
        if (num_outputs == 1) {
            oss << "return out0;";
            return;
        }
        oss << "return std::tuple";
        oss << output_type;
        oss << "{";
        for (int i=0; i < num_outputs; i++) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "out" + std::to_string(i);
        }
        oss << "};";
    }

    void CodeGenVisitor::visit(std::shared_ptr<CallStarRepeat> node) {
        oss << "([&]{\n";

        oss << "auto out = ";
        auto call = IR::make<Call>(node->function, node->arguments);
        call->accept(this);
        oss << ";\n";

        if (node->arguments.size() > 1) {
            oss << "auto& [";
            for(int i=0; i < node->arguments.size(); i++) {
                if (i > 0) {
                    oss << ", ";
                }
                auto var = "out" + std::to_string(i);
                oss << var;
            }
            oss << "] = out;\n";
        } else {
            oss << "auto& out0 = out;\n";
        }

        generate_for_loop("iter", IR::make<Literal>(node->numIterations - 1, Datatype::intType()));


        indent();
        oss << get_indent();
        if (node->arguments.size() == 1) {
            oss << "out0";
        } else {
            oss << "std::tie(";
            for(int i=0; i < node->arguments.size(); i++) {
                if (i > 0) {
                    oss << ", ";
                }
                auto var = "out" + std::to_string(i);
                oss << var;
            }
            oss << ")";
        }
        oss << " = ";
        auto args = std::vector<std::shared_ptr<Expression>>();
        for(int i=0; i < node->arguments.size(); i++) {
            args.push_back(IR::make<ReadAccess>("out" + std::to_string(i), false));
        }
        auto call_ = IR::make<Call>(node->function, args);
        call_->accept(this);
        oss << ";\n";

        unindent();

        oss << get_indent();
        oss << "}\n";
        get_lambda_return(node->getType()->dump(), node->arguments.size());
        oss << "\n}())";
    }

    void CodeGenVisitor::visit(std::shared_ptr<CallStarCondition> node) {
        oss << "([&]{\n";

        oss << "auto out = ";
        auto call = IR::make<Call>(node->function, node->arguments);
        call->accept(this);
        oss << ";\n";

        if (node->arguments.size() > 1) {
            oss << "auto& [";
            for(int i=0; i < node->arguments.size(); i++) {
                if (i > 0) {
                    oss << ", ";
                }
                auto var = "out" + std::to_string(i);
                oss << var;
            }
            oss << "] = out;\n";
        } else {
            oss << "auto& out0 = out;\n";
        }

        generate_while_loop(node->stopCondition);

        indent();
        oss << get_indent();
        if (node->arguments.size() == 1) {
            oss << "out0";
        } else {
            oss << "std::tie(";
            for(int i=0; i < node->arguments.size(); i++) {
                if (i > 0) {
                    oss << ", ";
                }
                auto var = "out" + std::to_string(i);
                oss << var;
            }
            oss << ")";
        }
        oss << " = ";
        auto args = std::vector<std::shared_ptr<Expression>>();
        for(int i=0; i < node->arguments.size(); i++) {
            args.push_back(IR::make<ReadAccess>("out" + std::to_string(i), false));
        }
        auto call_ = IR::make<Call>(node->function, args);
        call_->accept(this);
        oss << ";\n";

        unindent();

        oss << get_indent();
        oss << "}\n";

        //TODO: dump() is not good enough for tensor types with complex expressions as dimensions
        get_lambda_return(node->getType()->dump(), node->arguments.size());
        oss << "\n}())";
    }

    void CodeGenVisitor::visit(std::shared_ptr<Module> node) {}

    void CodeGenVisitor::visit(std::shared_ptr<Reduction> node) {
        auto i = node->reductionVar->getName();

        std::string init_var = "init_";
        init_var += i;
        oss << get_indent();
        oss << "auto ";
        oss << init_var;
        oss << " = ";
        node->reductionInit->accept(this);
        oss << ";\n";

        generate_for_loop(node->reductionVar->getName(), node->reductionVar->dimension);
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

        for(const auto & reduction : reductions) {
            reduction->accept(this);
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
                exp = reduce_expression(init_var, IR::make<ReadAccess>(next_var, false), red->reductionOp);
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

    void CodeGenVisitor::visit(std::shared_ptr<BinaryOp> node) {

        if (node->left->precedence > node->precedence) {
            oss << "(";
            node->left->accept(this);
            oss << ")";
        } else {
            node->left->accept(this);
        }
        oss << " ";
        oss << node->op->sign;
        oss << " ";
        if ((node->right->precedence > node->precedence) ||  (node->right->precedence == node->precedence && node->isAsymmetric)){
            oss << "(";
            node->right->accept(this);
            oss << ")";
        } else {
            node->right->accept(this);
        }
    }

    void CodeGenVisitor::visit(std::shared_ptr<UnaryOp> node) {
        oss << node->op->sign;
        oss << " ";
        node->expr->accept(this);
    }

    std::shared_ptr<Expression> CodeGenVisitor::reduce_expression(const std::string& init_var, std::shared_ptr<Expression> expr, const std::shared_ptr<Operator>& op) {
        std::shared_ptr<Expression> left = IR::make<ReadAccess>(init_var, false);
        return IR::make<BinaryOp>(left, std::move(expr), op, op->type);
    }

    void CodeGenVisitor::generate_for_loop(const std::string& var, const std::shared_ptr<Expression>& dim) {
        oss << get_indent();
        oss << "for(int ";
        oss <<  var;
        oss << "=0; ";
        oss <<  var;
        oss <<  "<";
        dim->accept(this);
        oss << "; "; oss << var; oss << "++) {\n";
    };

    void CodeGenVisitor::generate_while_loop(const std::shared_ptr<Expression>& condition) {
        oss << get_indent();
        oss << "while(!";
        oss << "(";
        condition->accept(this);
        oss << ")";
        oss <<  ") {\n";
    }
}