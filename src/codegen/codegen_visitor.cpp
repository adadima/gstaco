//
// Created by Alexandra Dima on 26.12.2021.
//

#include "einsum_taco/codegen/codegen_visitor.h"

#include<iostream>
#include <string_view>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace einsum {

    static std::string readFileIntoString(const std::string& path) {
        FILE *fp = fopen(path.c_str(), "r");
        if (fp == nullptr) {
            std::cout << "Failed to open file for reading " << path << std::endl;
            std::abort();
        }
        auto size = fs::file_size(path);
        std::string contents = std::string(size, 0);
        fread(contents.data(), 1, size, fp);
        fclose(fp);
        return contents;
    }

    std::string get_runtime_dir() {
        return {GSTACO_RUNTIME};
    }

    std::string parse_variable_name(const std::string& var) {
        if (var.rfind('#', 0) == 0) {
            auto nth = std::stoi(var.substr(1));
            return "out" + std::to_string(nth - 1);
        }
        return var;
    }

    void CodeGenVisitor::generate_tensor_template() {
        auto tensor_template = readFileIntoString(get_runtime_dir() + "tensor.h");
        *oss << tensor_template;
    }

    void CodeGenVisitor::visit(std::shared_ptr<IndexVar> node) {
        *oss << node->dump();
    }
    void CodeGenVisitor::visit(std::shared_ptr<Literal> node) {
        *oss << node->dump();
    }

    void CodeGenVisitor::visit(std::shared_ptr<IndexVarExpr> node) {
        node->indexVar->accept(this);
    }

    template<typename T>
    void CodeGenVisitor::visit_tensor_access(const std::shared_ptr<T>& access) {
        *oss << parse_variable_name(access->tensor->name);
        if (access->indices.size() == 0) {
            return;
        }
        *oss << ".at({";
        auto indices = access->indices;
        for (int i=0; i < indices.size(); i++) {
            if (i > 0) {
                *oss << ", ";
            }
            indices[i]->accept(this);
        }
        *oss << "})";
    }

    void CodeGenVisitor::visit(std::shared_ptr<Access> node) {
        visit_tensor_access(node);
    }

    void CodeGenVisitor::visit(std::shared_ptr<ReadAccess> node) {
        visit_tensor_access(node);
    }

    // TODO: generate asserts that index var dimensions match
    void CodeGenVisitor::visit(std::shared_ptr<Definition> node) {
        for(int a=0; a < node->lhs.size(); a++) {
            auto lhs = node->lhs[a];
            if (lhs->tensor->name == "_") {
                continue;
            }
            *oss << get_indent();
            *oss << "{\n";
            indent();

            for(auto &&acc : lhs->indices) {
                generate_for_loop(acc->getName(), acc->dimension);
                indent();
            }

            auto init_var = visit_reduced_expr(node->rhs, node->reduction_list);

            *oss << get_indent();
            lhs->accept(this);
            *oss << " = ";
            if (node->lhs.size() > 1) {
                *oss << "std::get<";
                *oss << std::to_string(a);
                *oss << ">(";
                *oss << init_var;
                *oss << ")";
            } else {
                *oss << init_var;
            }
            *oss << ";\n";

            for(auto &&acc: lhs->indices) {
                unindent();
                *oss << get_indent();
                *oss << "}\n";
            }
            unindent();
            *oss << get_indent();
            *oss << "}\n";
        }
    }

    void CodeGenVisitor::visit_func_signature(std::shared_ptr<FuncDecl> node) {
        auto return_type = node->getOutputType();
        return_type->accept(this);
        *oss << " ";
        *oss << node->funcName;
        *oss << "(";
        for (int i=0; i < node->inputs.size(); i++) {
            if (i > 0) {
                *oss << ", ";
            }
            auto input = node->inputs[i];
            input->getType()->accept(this);
            *oss << " ";
            *oss << input->name;
        }
        *oss << ")";
    }

    void CodeGenVisitor::visit(std::shared_ptr<FuncDecl> node) {
        visit_func_signature(node);
        *oss << " {\n";
        indent();
        for (auto &stmt: node->body) {
            stmt->accept(this);
            *oss << "\n";
        }
        unindent();
        std::vector<std::string> out_names= {};
        out_names.reserve(node->outputs.size());
        for (auto & output : node->outputs) {
            out_names.push_back(output->name);
        }
        print_return(node->getOutputType(), out_names);
        *oss << "\n}\n";
    }

    void CodeGenVisitor::visit(std::shared_ptr<Call> node) {
        *oss << node->function->funcName;
        *oss << "(";
        for (int i=0; i < node->arguments.size(); i++) {
            if (i > 0) {
                *oss << ", ";
            }
            node->arguments[i]->accept(this);
        }
        *oss << ")";
    }

    void CodeGenVisitor::get_lambda_return(const std::shared_ptr<TupleType>& output_type, int num_outputs) {
        std::vector<std::string> out_names= {};
        out_names.reserve(num_outputs);
        for (int i=0; i < num_outputs; i++) {
            out_names.push_back("out" + std::to_string(i));
        }
        print_return(output_type, out_names);
    }

    void CodeGenVisitor::print_return(const std::shared_ptr<TupleType>& output_type, const std::vector<std::string>& output_names) {
        if (output_names.size() == 1) {
            *oss << "return ";
            *oss << output_names[0];
            *oss << ";";
            return;
        }
        *oss << "return ";
        output_type->accept(this);
        *oss << "{";
        for (int i=0; i < output_names.size(); i++) {
            if (i > 0) {
                *oss << ", ";
            }
            *oss << output_names[i];
        }
        *oss << "};";
    }

    void CodeGenVisitor::visit_call(const std::shared_ptr<Call>& node, const std::function<void()>& loop_generator) {
        *oss << "([&]{\n";

        *oss << "auto out = ";
        auto call = IR::make<Call>(node->function, node->arguments);
        call->accept(this);
        *oss << ";\n";

        if (node->arguments.size() > 1) {
            *oss << "auto& [";
            for(int i=0; i < node->arguments.size(); i++) {
                if (i > 0) {
                    *oss << ", ";
                }
                auto var = "out" + std::to_string(i);
                *oss << var;
            }
            *oss << "] = out;\n";
        } else {
            *oss << "auto& out0 = out;\n";
        }

        loop_generator();

        indent();
        *oss << get_indent();
        if (node->arguments.size() == 1) {
            *oss << "out0";
        } else {
            *oss << "std::tie(";
            for(int i=0; i < node->arguments.size(); i++) {
                if (i > 0) {
                    *oss << ", ";
                }
                auto var = "out" + std::to_string(i);
                *oss << var;
            }
            *oss << ")";
        }
        *oss << " = ";
        auto args = std::vector<std::shared_ptr<Expression>>();
        for(int i=0; i < node->arguments.size(); i++) {
            args.push_back(IR::make<ReadAccess>("out" + std::to_string(i), false));
        }
        auto call_ = IR::make<Call>(node->function, args);
        call_->accept(this);
        *oss << ";\n";

        unindent();

        *oss << get_indent();
        *oss << "}\n";
        get_lambda_return(node->function->getOutputType(), node->arguments.size());
        *oss << "\n}())";
    }

    void CodeGenVisitor::visit(std::shared_ptr<CallStarRepeat> node) {
        visit_call(node, [&]() {
            generate_for_loop("iter", IR::make<Literal>(node->numIterations - 1, Datatype::intType()));
        });
    }

    void CodeGenVisitor::visit(std::shared_ptr<CallStarCondition> node) {
        visit_call(node, [&]() {
            generate_while_loop(node->stopCondition);
        });
    }

    void CodeGenVisitor::visit(std::shared_ptr<Module> node) {
        oss = oss_h;
        generate_tensor_template();
        oss = oss_cpp;

        *oss << "#include \"" << module_name << ".h\"" << std::endl;

        for(auto &comp: node->decls) {
            comp->accept(this);
            *oss << "\n";
            if (comp->is_decl()) {
                oss = oss_h;
                visit_func_signature(comp->as_decl());
                *oss << ";\n";
                oss = oss_cpp;
            }
        }
    }

    void CodeGenVisitor::visit(std::shared_ptr<Reduction> node) {
        auto i = node->reductionVar->getName();

        std::string init_var = "init_";
        init_var += i;
        *oss << get_indent();
        *oss << "auto ";
        *oss << init_var;
        *oss << " = ";
        node->reductionInit->accept(this);
        *oss << ";\n";

        generate_for_loop(node->reductionVar->getName(), node->reductionVar->dimension);
    }

    //TODO: add identifiers to variable names
    //TODO: make new rewriter passes to make lhs and rhs rank-0
    std::string CodeGenVisitor::visit_reduced_expr(const std::shared_ptr<Expression>& expr, const std::vector<std::shared_ptr<Reduction>>& reductions) {
        if (reductions.empty()) {
            *oss << get_indent();
            *oss << "auto init = ";
            expr->accept(this);
            *oss << ";\n";
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

            *oss << get_indent();
            *oss << init_var;
            *oss << " = ";
            exp->accept(this);
            *oss << ";\n";

            unindent();
            *oss << get_indent();
            *oss << "}\n";

            if (!r) {
                var = init_var;
            }
        }

        return var;
    }

    void CodeGenVisitor::visit(std::shared_ptr<BinaryOp> node) {

        if (node->left->precedence > node->precedence) {
            *oss << "(";
            node->left->accept(this);
            *oss << ")";
        } else {
            node->left->accept(this);
        }
        *oss << " ";
        *oss << node->op->sign;
        *oss << " ";
        if ((node->right->precedence > node->precedence) ||  (node->right->precedence == node->precedence && node->isAsymmetric)){
            *oss << "(";
            node->right->accept(this);
            *oss << ")";
        } else {
            node->right->accept(this);
        }
    }

    void CodeGenVisitor::visit(std::shared_ptr<UnaryOp> node) {
        *oss << node->op->sign;
        *oss << " ";
        node->expr->accept(this);
    }

    std::shared_ptr<Expression> CodeGenVisitor::reduce_expression(const std::string& init_var, std::shared_ptr<Expression> expr, const std::shared_ptr<Operator>& op) {
        std::shared_ptr<Expression> left = IR::make<ReadAccess>(init_var, false);
        return IR::make<BinaryOp>(left, std::move(expr), op, op->type);
    }

    void CodeGenVisitor::generate_for_loop(const std::string& var, const std::shared_ptr<Expression>& dim) {
        *oss << get_indent();
        *oss << "for(int ";
        *oss <<  var;
        *oss << "=0; ";
        *oss <<  var;
        *oss <<  "<";
        dim->accept(this);
        *oss << "; "; *oss << var; *oss << "++) {\n";
    };

    void CodeGenVisitor::generate_while_loop(const std::shared_ptr<Expression>& condition) {
        *oss << get_indent();
        *oss << "while(!";
        *oss << "(";
        condition->accept(this);
        *oss << ")";
        *oss <<  ") {\n";
    }

    void CodeGenVisitor::visit(std::shared_ptr<Datatype> node) {
        *oss << node->dump();
    }

    void CodeGenVisitor::visit(std::shared_ptr<TensorType> node) {
        if (node->getOrder() == 0) {
            node->getElementType()->accept(this);
            return;
        }
        *oss << "Tensor<";
        node->getElementType()->accept(this);
        *oss << ", ";
        *oss << std::to_string(node->getOrder());
        *oss << ">";
    }

    //TODO: make this a helper, not method on visitor. TupleType should not be a node
    void CodeGenVisitor::visit(std::shared_ptr<TupleType> node) {
        if (node->tuple.size() == 1) {
            node->tuple[0]->accept(this);
            return;
        }
        *oss << "std::tuple<";
        for (int i=0; i < node->tuple.size(); i++) {
            if (i > 0) {
                *oss << ", ";
            }
            node->tuple[i]->accept(this);
        }
        *oss << ">";
    }

    void CodeGenVisitor::visit(std::shared_ptr<Operator> node) {

    }

    void CodeGenVisitor::visit(std::shared_ptr<TensorVar> tensor) {}

    // TODO: also allocate the memory in here, separately from the Tensor constructor!
    void CodeGenVisitor::visit(std::shared_ptr<Allocate> node) {
        if (node->tensor->getOrder() > 0) {
            *oss << node->tensor->name << ".allocate();";
        }
    }

    // Something like: t1.data = t2.data
    void CodeGenVisitor::visit(std::shared_ptr<MemAssignment> node) {
    }

    //Tensor<int, 3> t({2, 2, 2});
    void CodeGenVisitor::visit(std::shared_ptr<Initialize> node) {
        auto tensor = node->tensor;
        tensor->getType()->accept(this);
        *oss << " " << tensor->name;
        if (tensor->getOrder() == 0) {
            *oss << " = " << node->tensor->getType()->getElementType()->dumpDefault() << ";";
            return;
        }
        *oss << "({";
        auto dims = tensor->getDimensions();

        for (int i=0; i < dims.size(); i++) {
            if (i > 0) {
                *oss << ", ";
            }
            dims[i]->accept(this);
        }
        *oss << "});\n";
    }
}