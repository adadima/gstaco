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
            auto lhs = node.lhs[0];
            for(auto &&acc : lhs->indices) {
                generate_for_loop(acc->getName(), acc->dimension);
                indent();
            }

            oss << get_indent();
            oss << "auto out = ";
            node.rhs->accept(this);
            oss << ";\n";

            for(int i=0; i < node.lhs.size(); i++) {
                oss << get_indent();
                node.lhs[i]->accept(this);
                oss << " = std::get<";
                oss << std::to_string(i);
                oss << ">(out);\n";
            }

            for(auto &&acc: lhs->indices) {
                unindent();
                oss << get_indent();
                oss << "}\n";
            }
        } else {
            auto lhs = node.lhs[0];
            for(auto &&acc : lhs->indices) {
                generate_for_loop(acc->getName(), acc->dimension);
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
    }

    void CodeGenVisitor::visit(const FuncDecl& node) {}

    void CodeGenVisitor::visit(const Call& node) {
        oss << node.dump();
    }

    void CodeGenVisitor::get_lambda_return(std::string output_type, int num_outputs) {
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

    void CodeGenVisitor::visit(const CallStarRepeat& node) {
        oss << "([&]{\n";

        oss << "auto out = ";
        auto call = new Call(node.function, node.arguments);
        call->accept(this);
        oss << ";\n";

        if (node.arguments.size() > 1) {
            oss << "auto& [";
            for(int i=0; i < node.arguments.size(); i++) {
                if (i > 0) {
                    oss << ", ";
                }
                auto var = "out" + std::to_string(i);
                oss << var;
            }
            oss << "] = out;\n";
        }

        generate_for_loop("iter", IR::make<Literal>(node.numIterations - 1, Datatype::intType()));


        indent();
        oss << get_indent();
        if (node.arguments.size() == 1) {
            oss << "out";
        } else {
            oss << "std::tie(";
            for(int i=0; i < node.arguments.size(); i++) {
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
        for(int i=0; i < node.arguments.size(); i++) {
            args.push_back(IR::make<ReadAccess>("out" + std::to_string(i)));
        }
        auto call_ = new Call(node.function, args);
        call_->accept(this);
        oss << ";\n";

        unindent();

        oss << get_indent();
        oss << "}\n";
        get_lambda_return(node.getType()->dump(), node.arguments.size());
        oss << "\n}())";
    }

    void CodeGenVisitor::visit(const CallStarCondition& node) {
        oss << "([&]{\n";

        oss << "auto out = ";
        auto call = new Call(node.function, node.arguments);
        call->accept(this);
        oss << ";\n";

        if (node.arguments.size() > 1) {
            oss << "auto& [";
            for(int i=0; i < node.arguments.size(); i++) {
                if (i > 0) {
                    oss << ", ";
                }
                auto var = "out" + std::to_string(i);
                oss << var;
            }
            oss << "] = out;";
        }

        generate_while_loop(node.stopCondition);

        indent();
        oss << get_indent();
        if (node.arguments.size() == 1) {
            oss << "out";
        } else {
            oss << "std::tie(";
            for(int i=0; i < node.arguments.size(); i++) {
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
        for(int i=0; i < node.arguments.size(); i++) {
            args.push_back(IR::make<ReadAccess>("out" + std::to_string(i)));
        }
        auto call_ = new Call(node.function, args);
        call_->accept(this);
        oss << ";\n";

        unindent();

        oss << get_indent();
        oss << "}\n";
        get_lambda_return(node.getType()->dump(), node.arguments.size());
        oss << "\n}())";
    }

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

        generate_for_loop(node.reductionVar->getName(), node.reductionVar->dimension);
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
        oss << "while(! ";
        oss << "(";
        condition->accept(this);
        oss << ")";
        oss <<  ") {\n";
    };
}