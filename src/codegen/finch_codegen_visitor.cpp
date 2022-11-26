//
// Created by Alexandra Dima on 30.10.2022.
//

//
// Created by Alexandra Dima on 26.12.2021.
//

#include "einsum_taco/codegen/finch_codegen_visitor.h"

#include<iostream>
#include <string_view>
#include <string>
#include <sstream>
#include <fstream>
#include <streambuf>
#include "julia.h"
#include "finch.h"
#include "einsum_taco/codegen/codegen_utils.h"

namespace einsum {
    FinchCodeGenVisitor::FinchCodeGenVisitor(std::ostream* oss_cpp, std::ostream* oss_h, std::ostream* oss_drive, std::string module_name, bool main) :  oss(oss_cpp), oss_cpp(oss_cpp), oss_h(oss_h), module_name(std::move(module_name)),
            indent_(0), oss_drive(oss_drive) {
        if (!is_active_jl_session) {
            finch_initialize();
            finch_initialize();
        }
        finch_eval("using Finch;"
                   "using RewriteTools;"
                   "using Finch.IndexNotation: or, choose;"
                   "using SparseArrays;"
                   "using Pkg;"
                   "Pkg.add(\"MatrixMarket\");"
                   "using MatrixMarket;"
        );
    }

    FinchCodeGenVisitor::~FinchCodeGenVisitor() {}

    void FinchCodeGenVisitor::visit(std::shared_ptr<Module> node) {
        auto mapper_v = FuncPtr2TensorArgsMapper();
        node->accept(&mapper_v);
        def2tensor_args = mapper_v.def2tensor_args;
        for(auto&[defid, args]: def2tensor_args) {
            std::cout << "DEF " << defid << ": ";
            for(auto& arg: args) {
                std::cout << arg->dump() << "\n";
            }
        }
        def2func_ptr = mapper_v.def2func_ptr;

        oss = oss_h;
        generate_runtime_header();
        *oss_h << "\n#ifdef __cplusplus\n"
                  "}\n";

        oss = oss_drive;
        *oss << "#include \"" << module_name << ".h\"\n";

        for(auto &comp: node->decls) {
            if (comp->is_init()) {
                comp->as_init()->tensor->getType()->accept(this);
                *oss << " " << comp->as_init()->tensor->name << ";\n";
            }
        }
        *oss << "\nint main() {\n";

        oss = oss_cpp;
        *oss << "#include \"" << module_name << ".h\"" << std::endl;
        generate_runtime_source();
        *oss << "\n";

        auto jl_init_v = JlFunctionInitializer(oss);
        node->accept(&jl_init_v);

        auto checker = NeedsFinchVisitor();
        node->accept(&checker);
        def2needs_finch = checker.def2needs_finch;

        *oss << "void compile() {\n";
        std::stringstream junk;
        auto finch_compile_v = FinchCompileVisitor(module_name, oss, &junk, def2tensor_args, def2needs_finch);
        finch_compile_v.visit(node);
        *oss << "}\n";

        for(auto &comp: node->decls) {
            if (comp->is_builtin()) {
                comp->accept(this);
            } else if (comp->is_decl()) {
                comp->accept(this);
                *oss << "\n";

                oss = oss_h;
                visit_func_signature(comp->as_decl());
                *oss << ";\n";
                oss = oss_cpp;
            } else if (comp->is_init()) {
                auto init = comp->as_init();
                auto tensor = init->tensor;

                oss = oss_h;
                *oss << "extern ";
                tensor->getType()->accept(this);
                *oss << " " << tensor->name << ";\n";

                oss = oss_drive;
                *oss << tensor->name << " = 0;\n";

                oss = oss_cpp;
            }
        }
        *oss_h << "#endif\n";
        *oss_drive << " return 0; }\n";
    }


    void FinchCodeGenVisitor::generate_runtime_header() const {
//        auto tensor_template = readFileIntoString(get_runtime_include_dir() + "runtime.h");
        std::string header = "#include <julia.h>\n"
                             "#include \"finch.h\"\n"
                             "#include <stdio.h>\n"
                             "#include <stdint.h>\n"
                             "#include <stdarg.h>\n"
                             "#include <tuple>\n"
                             "\n"
                             "#ifdef __cplusplus\n"
                             "extern \"C\" {\n"
                             "#endif\n"
                             "\n"
                             "struct Graph {\n"
                             "    jl_value_t* weights;\n"
                             "    jl_value_t* edges;\n"
                             "};\n"
                             "\n"
                             "int make_weights_and_edges(const char *graph_path, struct Graph* graph);\n"
                             "\n"
                             "void enter_finch();\n"
                             "\n"
                             "void exit_finch();\n"
                             "void compile();\n";
        *oss << header;
    }

    void FinchCodeGenVisitor::generate_runtime_source() const {
//        auto driver_code = readFileIntoString(get_runtime_src_dir() + "runtime.c");
        std::string code = "#include <stdio.h>\n"
                           "\n"
                           "\n"
                           "int make_weights_and_edges(const char* graph_path, struct Graph* graph) {\n"
                           "    char code1[1000];\n"
                           "    sprintf(code1, \"matrix = copy(transpose(MatrixMarket.mmread(\\\"%s\\\")))\\n\\\n"
                           "        (n, m) = size(matrix)\\n\\\n"
                           "        @assert n == m\\n\\\n"
                           "        nzval = ones(size(matrix.nzval, 1))\\n\\\n"
                           "        Finch.Fiber(\\n\\\n"
                           "                 Dense(n,\\n\\\n"
                           "                 SparseList(n, matrix.colptr, matrix.rowval,\\n\\\n"
                           "                 Element{0}(nzval))))\", graph_path);\n"
                           "    graph->edges = finch_eval(code1);\n"
                           "\n"
                           "    char code2[1000];\n"
                           "    sprintf(code2, \"matrix = copy(transpose(MatrixMarket.mmread(\\\"%s\\\")))\\n\\\n"
                           "        (n, m) = size(matrix)\\n\\\n"
                           "        @assert n == m\\n\\\n"
                           "        Finch.Fiber(\\n\\\n"
                           "                 Dense(n,\\n\\\n"
                           "                 SparseList(n, matrix.colptr, matrix.rowval,\\n\\\n"
                           "                 Element{0}(matrix.nzval))))\", graph_path);\n"
                           "    graph->weights = finch_eval(code2);\n"
                           "\n"
                           "    int* n = (int*) finch_exec(\"%s.lvl.I\", graph->edges);\n"
                           "    return *n;\n"
                           "}\n"
                           "\n"
                           "\n"
                           "void enter_finch() {\n"
                           "    finch_initialize();\n"
                           "\n"
                           "    jl_value_t* res = finch_eval(\"using RewriteTools\\n\\\n"
                           "    using Finch.IndexNotation\\n\\\n"
                           "    using SparseArrays\\n\\\n"
                           "     using MatrixMarket\\n\\\n"
                           "    \");\n"
                           "}\n"
                           "\n"
                           "\n"
                           "void exit_finch() {\n"
                           "    finch_finalize();\n"
                           "}";
        *oss << code;
    }

//    void FinchCodeGenVisitor::visit(std::shared_ptr<IndexVar> node) {
//        *oss << node->dump();
//    }
//
    void FinchCodeGenVisitor::visit(std::shared_ptr<Literal> node) {
        *oss << node->dump();
    }
//
//    void FinchCodeGenVisitor::visit(std::shared_ptr<IndexVarExpr> node) {
//        *oss << node->dump();
//    }


    void FinchCodeGenVisitor::visit_func_signature(std::shared_ptr<FuncDecl> node) {
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

    void FinchCodeGenVisitor::visit(std::shared_ptr<FuncDecl> node) {
        visit_func_signature(node);
        *oss << " {\n";
        indent();
        for (auto &stmt: node->body) {
            stmt->accept(this);
            *oss << ";\n";
        }
        unindent();
        std::vector<std::string> out_names= {};
        out_names.reserve(node->outputs.size());
        for (auto & output : node->outputs) {
            out_names.push_back(output->name);
        }
        print_return(node->getOutputType(), out_names);
        *oss << "}\n";
    }

    void FinchCodeGenVisitor::get_lambda_return(const std::shared_ptr<TupleType>& output_type, int num_outputs) {
        std::vector<std::string> out_names= {};
        out_names.reserve(num_outputs);
        for (int i=0; i < num_outputs; i++) {
            out_names.push_back("out" + std::to_string(i));
        }
        print_return(output_type, out_names);
    }

    void FinchCodeGenVisitor::print_return(const std::shared_ptr<TupleType>& output_type, const std::vector<std::string>& output_names) {
        if (output_names.size() == 1) {
            *oss << "return ";
            *oss << output_names[0];
            *oss << ";\n";
            return;
        }
        *oss << "return std::make_tuple(";
        for (int i=0; i < output_names.size(); i++) {
            if (i > 0) {
                *oss << ", ";
            }
            *oss << output_names[i];
        }
        *oss << ");\n";
    }

//    void FinchCodeGenVisitor::visit(std::shared_ptr<Reduction> node) {}

    void FinchCodeGenVisitor::generate_for_loop(const std::string& var, const std::shared_ptr<Expression>& dim) {
        *oss << get_indent();
        *oss << "for(int ";
        *oss <<  var;
        *oss << "=0; ";
        *oss <<  var;
        *oss <<  "<";
        dim->accept(this);
        *oss << "; "; *oss << var; *oss << "++) {\n";
    };

    void FinchCodeGenVisitor::generate_while_loop(const std::shared_ptr<CallStarCondition>& node) {
        *oss << get_indent();
        *oss << "while(!";
        *oss << "(";
        int def_id = node->condition_def->id;
        // need to extract the boolean value out of the jl_value_t*
        if (def2needs_finch.find(def_id) != def2needs_finch.end()) {
            *oss << "[";
            auto args = def2tensor_args[node->condition_def->id];
            for(size_t i=0; i < args.size(); i++) {
                if (i > 0) {
                    *oss << ", ";
                }
                *oss << "&" << args[i]->name;
            }
            *oss << "]() {\n";
            node->condition_def->accept(this);
            *oss << ";\n";
            *oss << "jl_value_t *val = finch_exec(\"%s.lvl.lvl.val\", " << node->condition_def->lhs[0]->tensor->name << ");\n";
            *oss << "int* data = (int*) jl_array_data(val);\n";
            *oss << "return data[0];\n";
            *oss << "}()";
        } else {
            node->condition_def->accept(this);
        }
        *oss << ")) {\n";
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<TensorType> node) {
        if (node->getOrder() == 0) {
            node->getElementType()->accept(this);
            return;
        }
        *oss << "jl_value_t*";
    }

    //TODO: make this a helper, not method on visitor. TupleType should not be a node
    void FinchCodeGenVisitor::visit(std::shared_ptr<TupleType> node) {
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

//    void FinchCodeGenVisitor::visit(std::shared_ptr<Operator> node) {}

    // TODO: also allocate the memory in here, separately from the Tensor constructor!
    // EXAMPLE: B = Finch.Fiber(Dense(N, Element{typemax(Int64), Int64}()))
    void FinchCodeGenVisitor::visit(std::shared_ptr<Allocate> node) {
        auto tensor = std::dynamic_pointer_cast<TensorVar>(node->tensor);
        if (!tensor || tensor->getOrder() == 0) {
            return;  // no need to allocate space for a scalar
        }
        *oss << "{\n";
        *oss << "char code[1000];\n";
        *oss << "sprintf(code, ";

        std::stringstream ss;
        for(size_t i=0; i < tensor->getDimensions().size(); i++) {
            ss << "N" << i << " = %d\\n\\\n";
        }
        ss << "Finch.Fiber(\\n\\\n";
        for(size_t i=0; i < tensor->getDimensions().size(); i++) {
            if (tensor->type->formats[i]->format == Format::Dense) {
                ss << "Dense(N" << i << ",\\n\\\n";
            } else if (tensor->type->formats[i]->format == Format::Sparse) {
                ss << "SparseList(N" << i << ", [1,1], Int64[],\\n\\\n";
            }
        }
        ss << "Element{0,";
        ss << fdump(tensor->type->getElementType());
        ss << "}()";
        for(size_t i=0; i < tensor->getDimensions().size(); i++) {
            ss << ")";
        }
        ss << ")";
        auto s = ss.str();
        *oss << "\"" << s << "\"";
        for(size_t i=0; i < tensor->getDimensions().size(); i++) {
            *oss << ", ";
            tensor->getDimensions()[i]->accept(this);
        }
        *oss << ");\n";
        *oss << tensor->name << " = finch_eval(code);\n";
        *oss << "}\n";
    }

    // Something like: t1.data = t2.data
//    void FinchCodeGenVisitor::visit(std::shared_ptr<MemAssignment> node) {}

    void FinchCodeGenVisitor::visit(std::shared_ptr<Initialize> node) {
        node->tensor->type->accept(this);
        *oss << " " << node->tensor->name << ";\n";
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<TensorVar> node) {
//        assert(node->getOrder() == 0);
        *oss << node->name;
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<Access> node) {
//        assert(node->tensor->getOrder() == 0);
        *oss << node->tensor->name;
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<ReadAccess> node) {
        std::string varname = node->tensor->name;
        if (node->indices.size() == 0) {
            *oss << varname;
            return;
        }
//        jl_value_t *F_val = finch_exec("%s.lvl.lvl.val", F);
//        double *F_data = jl_array_data(F_val);
//        for(int i = 0; i < N; i++) {
//            if (F_data[i] != 0) {
//                return 0;
//            }
//        }
        // TODO: use jl_array_data to get C array and index into it
        throw std::runtime_error("TODO: use jl_array_data to get C array and index into it");
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<TupleVarReadAccess> node) {
        *oss << "std::get<" << node->idx << ">(" << node->var->name << ")";
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<BinaryOp> node) {
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

    void FinchCodeGenVisitor::visit(std::shared_ptr<UnaryOp> node) {
        *oss << node->op->sign;
        *oss << " ";
        node->expr->accept(this);
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->lhs->type->accept(this);
        *oss << " " << node->lhs->name << " = ";
        node->rhs->accept(this);
        *oss << ";";
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<Definition> node) {
        if (node->skip_codegen) {
            return;
        }
        std::cout << "CODEGEN ON: " << node->dump() << "WITH FINCH? " << def2needs_finch[node->id] << "\n";
        auto type = node->rhs->getType();
        if (std::dynamic_pointer_cast<Call>(node->rhs)) {
            for (int a=0; a < node->lhs.size(); a++) {
                if (node->lhs[a]->tensor->name == "_") {
                    continue;
                }
                node->lhs[a]->accept(this);
                *oss << " = ";

                if (node->lhs.size() > 1) {
                    *oss << "std::get<";
                    *oss << std::to_string(a);
                    *oss << ">(";
                    node->rhs->accept(this);
                    *oss << ")";
                } else {
                    node->rhs->accept(this);
                }

                *oss << "\n";
            }
            return;
        }
        if (!def2needs_finch[node->id]) {
            node->lhs[0]->accept(this);
            *oss << " = ";
            node->rhs->accept(this);
            return;
        }

        for (auto& acc: node->lhs) {
            if (acc->tensor->name == "_") {
                continue;
            }
            std::string func_name = def2func_ptr[node->id];
            std::vector<std::shared_ptr<TensorVar>> tensor_args = def2tensor_args[node->id];
            *oss << "finch_call(" << func_name;
            for(auto& tensor: tensor_args) {
                *oss << ", ";
                if (tensor->getOrder() == 0) {
                    *oss << "finch_" << fdump(tensor->getType()->getElementType()) << "(" << tensor->name << ")";
                } else {
                    *oss << tensor->name;
                }
            }
            *oss << ")\n";
        }
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<AddOperator> node) {
        oss = oss_h;
        auto signature = R"(template<typename T>
T )";
        auto op_func = R"((T left, T right) {
    return left + right;
}
)";
        *oss << signature << node->funcName << op_func;
        oss = oss_cpp;
    }
//
    void FinchCodeGenVisitor::visit(std::shared_ptr<MulOperator> node) {
        oss = oss_h;
        auto signature = R"(template<typename T>
T )";
        auto op_func = R"((T left, T right) {
    return left * right;
}
)";
        *oss << signature << node->funcName << op_func;
        oss = oss_cpp;
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<AndOperator> node) {
        auto signature = R"(bool )";
        auto op_func = R"((bool left, bool right) {
    return left && right;
}
)";
        *oss << signature << node->funcName << op_func;
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<OrOperator> node) {
        auto signature = R"(bool )";
        auto op_func = R"((bool left, bool right) {
    return left || right;
}
)";
        *oss << signature << node->funcName << op_func;
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<MinOperator> node) {
        oss = oss_h;
        auto signature = R"(template<typename T>
T )";
        auto op_func = R"((T left, T right) {
    if (left > right) {
        return right;
    } else {
        return left;
    }
}
)";
        *oss << signature << node->funcName << op_func;
        oss = oss_cpp;
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<ChooseOperator> node) {
        oss = oss_h;
        auto signature = R"(template<typename T>
T )";
        auto op_func = R"((T left, T right) {
    if (right) {
        return right;
    } else {
        return left;
    }
}
)";
        *oss << signature << node->funcName << op_func;
        oss = oss_cpp;
    }
//
    void FinchCodeGenVisitor::visit(std::shared_ptr<Call> node) {
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

    void FinchCodeGenVisitor::visit_call(const std::shared_ptr<Call>& node, const std::function<void()>& loop_generator) {
        *oss << "([&]{\n";

        *oss << "auto out = ";
        auto call = IR::make<Call>(node->function, node->arguments);
        call->accept(this);
        *oss << ";\n";

        if (node->arguments.size() > 1) {
            for(int i=0; i < node->arguments.size(); i++) {
                auto var = "out" + std::to_string(i);
                *oss << "auto " << var << " = std::get<" << i << ">(out);\n";
            }
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

    void FinchCodeGenVisitor::visit(std::shared_ptr<CallStarRepeat> node) {
        visit_call(node, [&]() {
            generate_for_loop("iter", IR::make<Literal>(node->numIterations - 1, Datatype::intType()));
        });
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<CallStarCondition> node) {
        visit_call(node, [&]() {
            generate_while_loop(node);
        });
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<Datatype> node) {
        *oss << node->dump();
    }


    void DefinitionVisitor::visit(std::shared_ptr<Definition> node) {
        for (auto& acc: node->lhs) {
            acc->accept(this);
        }

        node->rhs->accept(this);
    }

    void DefinitionVisitor::visit(std::shared_ptr<Access> node) {
        node->tensor->accept(this);
    }

    void DefinitionVisitor::visit(std::shared_ptr<ReadAccess> node) {
        node->tensor->accept(this);

        for(auto& idx: node->indices) {
            idx->accept(this);
        }
    }

    void DefinitionVisitor::visit(std::shared_ptr<IndexVarExpr> node) {}

    void DefinitionVisitor::visit(std::shared_ptr<BinaryOp> node) {
        node->left->accept(this);
        node->right->accept(this);
    }

    void DefinitionVisitor::visit(std::shared_ptr<UnaryOp> node) {
        node->expr->accept(this);
    }

    void DefinitionVisitor::visit(std::shared_ptr<Call> node) {
        for(auto& inp: node->arguments) {
            inp->accept(this);
        }
    }

    void DefinitionVisitor::visit(std::shared_ptr<TupleVarReadAccess> node) {
        // this shouldn't do anything
    }

    //  TODO: update to generate any virtual type, not just dense
    void DefinitionVisitor::visit(std::shared_ptr<TensorVar> node) {
        tensors.insert(node->name);
        *oss << "t_" << node->name << " = ";

        // tensor is zero dimensional => scalar
        if (node->getOrder() == 0) {
            *oss << fdump(node->type->getElementType()) << "\n";
        } else {
            *oss << "typeof(@fiber ";
            for (size_t i=0; i < node->getOrder(); i++) {
                if(node->type->formats[i]->format == Format::Dense) {
                    *oss << "d(";
                } else if(node->type->formats[i]->format == Format::Sparse) {
                    *oss << "sl(";
                }
            }
            *oss << "e(0)";
            for (size_t i=0; i < node->getOrder(); i++) {
                *oss << ")";
            }
            *oss << ")";
        }

        *oss << "\n";
        *oss << node->name << " = " << "Finch.virtualize(:" << node->name << ", t_" << node->name << ", ctx_2)\n";
    }

    // TODO: maybe implement these if there's time
    void DefinitionVisitor::visit(std::shared_ptr<CallStarRepeat> node) {
        for(auto& inp: node->arguments) {
            inp->accept(this);
        }
    }

    void DefinitionVisitor::visit(std::shared_ptr<CallStarCondition> node) {
        for(auto& inp: node->arguments) {
            inp->accept(this);
        }
        node->stopCondition->accept(this);
    }

    void DefinitionVisitor::visit(std::shared_ptr<Literal> node) {}

    void DefinitionVisitor::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->rhs->accept(this);
    }

    void JlFunctionInitializer::visit(std::shared_ptr<Definition> node) {
        for (auto& acc: node->lhs) {
            *oss << "jl_function_t* finch_def_code" << node->id << ";\n";
        }
        node->rhs->accept(this);
    }

    void JlFunctionInitializer::visit(std::shared_ptr<FuncDecl> node) {
        for(auto& stmt: node->body) {
            if (stmt->is_def()) {
                stmt->as_def()->accept(this);
            } else if (stmt->is_multi_def()) {
                stmt->as_multi_def()->accept(this);
            }
        }
    }

    void JlFunctionInitializer::visit(std::shared_ptr<Module> node) {
        for(auto& comp: node->decls) {
            if (comp->is_def()) {
                comp->as_def()->accept(this);
            } else if (comp->is_multi_def()) {
                comp->as_multi_def()->accept(this);
            }
            if (comp->is_decl() && !comp->is_builtin()) {
                comp->as_decl()->accept(this);
            }
        }
    }

    void JlFunctionInitializer::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->rhs->accept(this);
    }

    void FinchCompileVisitor::visit(std::shared_ptr<Access> node) {
        *oss << node->tensor->name << "[";
        for (size_t i=0; i < node->indices.size(); i++) {
            *oss << node->indices[i]->dump();
            if (i != node->indices.size() - 1) {
                *oss << ",";
            }
        }
        *oss << "]";
    }

    void FinchCompileVisitor::visit(std::shared_ptr<ReadAccess> node) {
        if (node->tensor->getOrder() == 0) {
            *oss << "$" + node->tensor->name;
            return;
        }
        *oss << node->tensor->name << "[";
        for (size_t i=0; i < node->indices.size(); i++) {
            node->indices[i]->accept(this);
            if (i != node->indices.size() - 1) {
                *oss << ",";
            }
        }
        *oss << "]";
    }

    void FinchCompileVisitor::visit(std::shared_ptr<IndexVarExpr> node) {
        *oss << node->dump();
    }

    void FinchCompileVisitor::visit(std::shared_ptr<Literal> node) {
        *oss << node->dump();
    }

    void FinchCompileVisitor::visit(std::shared_ptr<Definition> node) {
        oss = finch;
        if (!def2needs_finch[node->id]) {
            std::cout << "NO FINCH NEEDED: " << node->dump() << "\n";
            oss = junk;
            node->rhs->accept(this);
            return;
        }
        std::cout << "FINCH NEEDED FOR DEF " << node->id << ": " << node->dump() << "\n";
        auto old_oss = oss;
        for (auto& acc: node->lhs) {
            if (acc->tensor->name == "_") {
                continue;
            }
            std::stringstream ss;
            oss = &ss;
            ss << "ctx = Finch.LowerJulia()\ncode = Finch.contain(ctx) do ctx_2\n";
            auto v = DefinitionVisitor(&ss, def2needs_finch);
            v.visit(node);
            for (size_t i=0; i < node->reduction_list.size(); i++) {
                ss << "w_" << i << " = Finch.virtualize(:w_" << i << ", typeof(Scalar{0, ";
                acc->tensor->getType()->getElementType()->accept(this);
                ss << "}()), ctx_2, :w_" << i << ")\n";
            }
//            kernel = @finch_program (@loop i (@loop k A[i,k] = w1[] where (@loop j w1[] *= w2[] where (@loop l w2[] += B[i,j,k,l] * C[j] * D[l])) ))

            ss << "kernel = @finch_program ";
            for (auto& idx: acc->indices) {
                ss << "(@loop " << idx->name << " ";
            }

            // TODO: replace dump() with fdump()
            // TODO: make sure non-zero init values work
            acc->accept(this);
            *oss << " = ";
            if (node->reduction_list.size() > 0) {
                for (size_t i=0; i < node->reduction_list.size() - 1; i++) {
                    auto& red = node->reduction_list[i];
                    ss << "w_" << i << "[] where (w_" << i << "[] ";
                    red->reductionOp->accept(this);
                    *oss << "= ";
                }

                size_t id = node->reduction_list.size() - 1;
                auto& red = node->reduction_list[id];
                ss << "w_" << id << "[] where";
                for (auto& idx: node->getReductionVars()) {
                    ss << " (@loop " << idx;
                }
                ss << " w_" << id << "[] ";
                red->reductionOp->accept(this);
                *oss << "= ";
            }

            node->rhs->accept(this);

            if (node->reduction_list.size() > 0) {
                for (size_t i=0; i < 2 * node->reduction_list.size() - 1; i++) {
                    ss << ")";
                }
            }

            for (auto& idx: acc->indices) {
                ss << ")";
            }
            ss << "\n";
            ss << "kernel_code = Finch.execute_code_virtualized(kernel, ctx_2)\n";
            ss << "end\n";
            ss << "return quote\n";
            ss << "     function def_" << node->id << "_" << module << "(";
            for (auto curr = def2args[node->id].begin();curr != def2args[node->id].end();) {
                ss << (*curr)->name;
                if (++curr != def2args[node->id].end()) {
                    ss << ", ";
                }
            }
            ss << ")\n";
            for(size_t i=0; i < node->reduction_list.size(); i++) {
                ss << "         w_" << i << " = Scalar{0}()\n";
            }
            ss << "         $code\n";
            ss << "     end\n";
            ss << "end\n";
            std::string virtualized =ss.str();
            const char* vc = virtualized.data();
            std::cout << vc;
            jl_value_t* expr = finch_exec(vc);
            std::cout << "Compiled finch\n";
            jl_value_t* code = finch_exec("string(last(%s.args))", expr);
            auto s = jl_string_data(code);

            *old_oss << "finch_def_code" << node->id << " = finch_eval(\n";

            std::stringstream ss_(s);
            std::string line;
            while(std::getline(ss_, line, '\n')){
                *old_oss << "\"" << line << "\\n\"\n";
            }
            *old_oss << ");\n";
        }

        oss = old_oss;
    }

    void FinchCompileVisitor::visit(std::shared_ptr<MinOperator> node) {
        *oss << "<<min>>";
    }

    void FinchCompileVisitor::visit(std::shared_ptr<ChooseOperator> node) {
        *oss << "<<choose>>";
    }

    void FinchCompileVisitor::visit(std::shared_ptr<AddOperator> node) {
        *oss << node->op->sign;
    }

    void FinchCompileVisitor::visit(std::shared_ptr<MulOperator> node) {
        *oss << node->op->sign;
    }

    void FinchCompileVisitor::visit(std::shared_ptr<AndOperator> node) {
        *oss << node->op->sign;
    }

    void FinchCompileVisitor::visit(std::shared_ptr<OrOperator> node) {
        *oss << "<<or>>";
    }

    void FinchCompileVisitor::visit(std::shared_ptr<TensorVar> tensor) {
        *oss << tensor->name;
    }

    void FinchCompileVisitor::visit(std::shared_ptr<Datatype> node) {
        *oss << fdump(node);
    }

    void FinchCompileVisitor::visit(std::shared_ptr<BinaryOp> node) {
        bool paren_left = node->left->precedence > node->precedence;
        bool paren_right = (node->right->precedence > node->precedence) ||  (node->right->precedence == node->precedence && node->isAsymmetric);

        if (paren_left) {
            *oss << "(";
        }
        node->left->accept(this);
        if (paren_left) {
            *oss << ")";
        }

        *oss << " " << node->op->sign  << " ";

        if (paren_right) {
            *oss << "(";
        }
        node->right->accept(this);
        if (paren_right) {
            *oss << ")";
        }
    }

    void FinchCompileVisitor::visit(std::shared_ptr<UnaryOp> node) {
        *oss << node->op->sign;
        node->expr->accept(this);
    }

    void FinchCompileVisitor::visit(std::shared_ptr<Call> node) {
        *oss << node->function->funcName << "(";
        for (int i=0; i < node->arguments.size(); i++) {
            *oss << node->arguments[i]->dump();
            if (i < node->arguments.size() - 1) {
                *oss << ", ";
            }
        }
        *oss << ")";
    }

    void FinchCompileVisitor::visit(std::shared_ptr<CallStarCondition> node) {
        std::cout << "GENERATING FINCH FOR WHILE CONDITION\n";
        node->condition_def->accept(this);
    }

    void FinchCompileVisitor::visit(std::shared_ptr<Module> node) {
        for(auto& decl: node->decls) {
            if (decl->is_decl() && !decl->is_builtin()) {
                decl->as_decl()->accept(this);
            } else if (decl->is_multi_def()) {
                decl->as_multi_def()->accept(this);
            } else if (decl->is_def()) {
                decl->as_def()->accept(this);
            }
        }
    }

    void FinchCompileVisitor::visit(std::shared_ptr<FuncDecl> node) {
        for(auto& stmt: node->body) {
            if (stmt->is_def()) {
                stmt->as_def()->accept(this);
            } else if (stmt->is_multi_def()) {
                stmt->as_multi_def()->accept(this);
            }
            else {
                std::cout << "NOT DEF: " << stmt->dump() << "\n";
                std::cout << "TYPE: " << typeid(stmt).name() << "\n";
            }
        }
    }

    void FinchCompileVisitor::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->rhs->accept(this);
    }

    void FinchCompileVisitor::visit(std::shared_ptr<CallStarRepeat> node) {}

    std::string fdump(std::shared_ptr<Datatype> node) {
        switch (node->getKind()) {
            case Datatype::Kind::Bool:
                return "Bool";
            case Datatype::Kind::Int:
                return "Int64";
            case Datatype::Kind::Float:
                return "Float64";
        }
    }

    void TensorCollector::visit(std::shared_ptr<Definition> node) {
        tensors.clear();
        for (auto& acc: node->lhs) {
            acc->tensor->accept(this);
        }
        node->rhs->accept(this);
    }

    void TensorCollector::visit(std::shared_ptr<ReadAccess> node) {
        node->tensor->accept(this);
        for (auto& idx: node->indices) {
            idx->accept(this);
        }
    }

    void TensorCollector::visit(std::shared_ptr<TupleVarReadAccess> node) {
        // don;t do anything here.
    }

    void TensorCollector::visit(std::shared_ptr<IndexVarExpr> node) {}



    void TensorCollector::visit(std::shared_ptr<BinaryOp> node) {
        node->right->accept(this);
        node->left->accept(this);
    }

    void TensorCollector::visit(std::shared_ptr<UnaryOp> node) {
        node->expr->accept(this);
    }

    void TensorCollector::visit_call(std::shared_ptr<Call> node) {
        for(auto& arg: node->arguments) {
            arg->accept(this);
        }
    }

    void TensorCollector::visit(std::shared_ptr<Call> node) {
        visit_call(node);
    }

    void TensorCollector::visit(std::shared_ptr<CallStarRepeat> node) {
        visit_call(node);
    }

    void TensorCollector::visit(std::shared_ptr<CallStarCondition> node) {
        visit_call(node);
        node->condition_def->accept(this);
    }

    void TensorCollector::visit(std::shared_ptr<TensorVar> node) {
        std::cout << "COLLECTING TENSORS FROM VAR: " << node->dump() << "\n";
        std::string tensor = node->name;
        if (seen.find(tensor) == seen.end()) {
            std::cout << tensor << "\n";
            seen.insert(tensor);
            auto new_name = tensor;
            if (new_name != tensor) {
                tensors.push_back(IR::make<TensorVar>(new_name, node->type, node->is_global));
            } else {
                tensors.push_back(node);
            }
        }
    }

    void TensorCollector::visit(std::shared_ptr<Literal> node) {
//        std::cout << "LITERAL: " << node->dump() << "\n";
    }

    void TensorCollector::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->rhs->accept(this);
    }

    void FuncPtr2TensorArgsMapper::visit(std::shared_ptr<Definition> node) {
        std::cout << "FuncPtr2TensorArgsMapper: " << node->dump() << "\n";
        auto v = TensorCollector();
        node->rhs->accept(&v);
        std::cout << "v.tensors size INITIALLY: " << v.tensors.size() << "\n";
        for (auto& acc: node->lhs) {
            auto tensors = v.tensors;
            tensors.push_back(acc->tensor);
            std::cout << "v.tensors size: " << v.tensors.size() << "\n";
            std::cout << "tensors.size: " << tensors.size() << "\n";
            char func_name[100];
            sprintf(func_name, "finch_def_code%d", node->id);
            def2tensor_args.insert({node->id, tensors});
            def2func_ptr.insert({node->id, func_name});
        }
        node->rhs->accept(this);
    }

    void FuncPtr2TensorArgsMapper::visit(std::shared_ptr<FuncDecl> node) {
        for(auto& stmt: node->body) {
            if (stmt->is_def()) {
                stmt->as_def()->accept(this);
            } else if (stmt->is_multi_def()) {
                stmt->as_multi_def()->accept(this);
            }
        }
    }

    void FuncPtr2TensorArgsMapper::visit(std::shared_ptr<Module> node) {
        for(auto& comp: node->decls) {
            if (comp->is_def()) {
                comp->as_def()->accept(this);
            }
            else if (comp->is_multi_def()) {
                comp->as_multi_def()->accept(this);
            }
            else if (comp->is_decl() && !comp->is_builtin()) {
                comp->as_decl()->accept(this);
            }
        }
    }

    void FuncPtr2TensorArgsMapper::visit(std::shared_ptr<CallStarCondition> node) {
        std::cout << "FuncPtr2TensorArgsMapper: CALLSTAR";
        node->condition_def->accept(this);
    }

    void FuncPtr2TensorArgsMapper::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->rhs->accept(this);
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<Definition> node) {
        if (!node->reduction_list.empty()) {
            needs_finch = true;
        }
        for (auto& acc: node->lhs) {
            acc->accept(this);
        }
        node->rhs->accept(this);

    }

    void FinchDefinitionChecker::visit(std::shared_ptr<Literal> node) {
        needs_finch |= false; // stays the same, but it's clearer to show why
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<IndexVarExpr> node) {
        needs_finch = true;
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<Access> node) {
        needs_finch |= (!node->indices.empty());
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<ReadAccess> node) {
        needs_finch |= !(node->indices.empty());
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<TupleVarReadAccess> node) {
        needs_finch |= false; // stays the same, but it's clearer to show why
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<CallStarCondition> node) {
        visit_call(node);
        node->condition_def->accept(this);
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<CallStarRepeat> node) {
        visit_call(node);
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<Call> node) {
        visit_call(node);
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<TensorVar> node) {
        needs_finch |= false; // stays the same, but it's clearer to show why
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<BinaryOp> node) {
        node->left->accept(this);
        node->right->accept(this);
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<UnaryOp> node) {
        node->expr->accept(this);
    }

    void FinchDefinitionChecker::visit_call(std::shared_ptr<Call> node) {
        for(auto& arg: node->arguments) {
            arg->accept(this);
        }
    }

    void FinchDefinitionChecker::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->rhs->accept(this);
    }

    void NeedsFinchVisitor::visit(std::shared_ptr<Definition> node) {
        if (std::dynamic_pointer_cast<CallStarCondition>(node->rhs) == nullptr) {
            auto checker = FinchDefinitionChecker();
            node->accept(&checker);
            def2needs_finch.insert({node->id, checker.needs_finch});
        }
        node->rhs->accept(this);
    }
}