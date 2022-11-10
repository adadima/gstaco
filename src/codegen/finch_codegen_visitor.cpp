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
    FinchCodeGenVisitor::FinchCodeGenVisitor(std::ostream* oss_cpp, std::ostream* oss_h, std::ostream* oss_finch, std::string module_name, bool main) : oss(oss_cpp), oss_cpp(oss_cpp), oss_h(oss_h), module_name(std::move(module_name)),
            indent_(0), oss_finch(oss_finch) {
        finch_compile = new std::stringstream();
        finch_initialize();
        finch_eval("using Pkg;"
                   "using Finch;"
                   "using RewriteTools;"
                   "using Finch.IndexNotation: or_, choose;"
                   "using SparseArrays;"
                   "Pkg.add(\"MatrixMarket\");"
                   "using MatrixMarket;");
    }

    FinchCodeGenVisitor::~FinchCodeGenVisitor() {
        delete finch_compile;
        finch_finalize();
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<Module> node) {
        oss = oss_h;
        generate_runtime_header();

        oss = oss_cpp;
        *oss << "#include \"" << module_name << ".h\"" << std::endl;
        generate_runtime_source();
        *oss << "\n";

        std::cout << "IN CODEGEN\n";
        auto jl_init_v = JlFunctionInitializer(oss);
        jl_init_v.visit(node);

        std::cout << "AFTER JL INIT\n";
        *oss << "void compile() {\n";
        auto finch_compile_v = FinchCompileVisitor(oss);
        finch_compile_v.visit(node);
        *oss << "}\n";
        std::cout << "AFTER FINCH COMPILE\n";

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

                oss = oss_cpp;
            }
        }
        *oss_h << "#ifdef __cplusplus\n"
                 "}\n"
                 "#endif\n";
    }


    void FinchCodeGenVisitor::generate_runtime_header() const {
//        auto tensor_template = readFileIntoString(get_runtime_include_dir() + "runtime.h");
        std::string header = "#include <julia.h>\n"
                             "#include \"finch.h\"\n"
                             "#include <stdio.h>\n"
                             "#include <stdint.h>\n"
                             "#include <stdarg.h>\n"
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
                             "int make_weights_and_edges(const char *graph_name, struct Graph* graph);\n"
                             "\n"
                             "void enter_finch();\n"
                             "\n"
                             "void exit_finch();\n";
        *oss << header;
    }

    void FinchCodeGenVisitor::generate_runtime_source() const {
//        auto driver_code = readFileIntoString(get_runtime_src_dir() + "runtime.c");
        std::string code = "#include <stdio.h>\n"
                           "\n"
                           "\n"
                           "int make_weights_and_edges(const char* graph_name, struct Graph* graph) {\n"
                           "    char code1[1000];\n"
                           "    sprintf(code1, \"matrix = copy(transpose(MatrixMarket.mmread(\\\"./graphs/%s\\\")))\\n\\\n"
                           "        (n, m) = size(matrix)\\n\\\n"
                           "        @assert n == m\\n\\\n"
                           "        nzval = ones(size(matrix.nzval, 1))\\n\\\n"
                           "        Finch.Fiber(\\n\\\n"
                           "                 Dense(n,\\n\\\n"
                           "                 SparseList(n, matrix.colptr, matrix.rowval,\\n\\\n"
                           "                 Element{0}(nzval))))\", graph_name);\n"
                           "    graph->edges = finch_eval(code1);\n"
                           "    printf(\"Loaded edges\\n\");\n"
                           "\n"
                           "    char code2[1000];\n"
                           "    sprintf(code2, \"matrix = copy(transpose(MatrixMarket.mmread(\\\"./graphs/%s\\\")))\\n\\\n"
                           "        (n, m) = size(matrix)\\n\\\n"
                           "        @assert n == m\\n\\\n"
                           "        Finch.Fiber(\\n\\\n"
                           "                 Dense(n,\\n\\\n"
                           "                 SparseList(n, matrix.colptr, matrix.rowval,\\n\\\n"
                           "                 Element{0}(matrix.nzval))))\", graph_name);\n"
                           "    graph->weights = finch_eval(code2);\n"
                           "    printf(\"Loaded weights\\n\");\n"
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

    void FinchCodeGenVisitor::visit(std::shared_ptr<IndexVar> node) {
        *oss << node->dump();
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<Literal> node) {
        *oss << node->dump();
    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<IndexVarExpr> node) {
        *oss << node->dump();
    }


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

    void FinchCodeGenVisitor::get_lambda_return(const std::shared_ptr<TupleType>& output_type, int num_outputs) {}

    void FinchCodeGenVisitor::print_return(const std::shared_ptr<TupleType>& output_type, const std::vector<std::string>& output_names) {}

    void FinchCodeGenVisitor::visit_call(const std::shared_ptr<Call>& node, const std::function<void()>& loop_generator) {}

    void FinchCodeGenVisitor::visit(std::shared_ptr<CallStarRepeat> node) {}

    void FinchCodeGenVisitor::visit(std::shared_ptr<CallStarCondition> node) {}

    void FinchCodeGenVisitor::visit(std::shared_ptr<Reduction> node) {}

    void FinchCodeGenVisitor::generate_for_loop(const std::string& var, const std::shared_ptr<Expression>& dim) {};

    void FinchCodeGenVisitor::generate_while_loop(const std::shared_ptr<Expression>& condition) {}

    void FinchCodeGenVisitor::visit(std::shared_ptr<TensorType> node) {}

    //TODO: make this a helper, not method on visitor. TupleType should not be a node
    void FinchCodeGenVisitor::visit(std::shared_ptr<TupleType> node) {}

    void FinchCodeGenVisitor::visit(std::shared_ptr<Operator> node) {}

    // TODO: also allocate the memory in here, separately from the Tensor constructor!
    void FinchCodeGenVisitor::visit(std::shared_ptr<Allocate> node) {}

    // Something like: t1.data = t2.data
    void FinchCodeGenVisitor::visit(std::shared_ptr<MemAssignment> node) {}

    void FinchCodeGenVisitor::visit(std::shared_ptr<Initialize> node) {}

    void FinchCodeGenVisitor::visit(std::shared_ptr<TensorVar> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<Access> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<ReadAccess> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<BinaryOp> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<UnaryOp> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<Definition> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<AddOperator> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<MulOperator> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<AndOperator> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<OrOperator> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<MinOperator> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<ChooseOperator> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<Call> node) {

    }

    void FinchCodeGenVisitor::visit(std::shared_ptr<Datatype> node) {

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
    }

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
                *oss << "d(";
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
    void DefinitionVisitor::visit(std::shared_ptr<CallStarRepeat> node) {}

    void DefinitionVisitor::visit(std::shared_ptr<CallStarCondition> node) {}

    void JlFunctionInitializer::visit(std::shared_ptr<Definition> node) {
        for (auto& acc: node->lhs) {
            *oss << "jl_function_t* finch_def_code_" << def_id << ";\n";
            def_id += 1;
        }
    }

    void JlFunctionInitializer::visit(std::shared_ptr<FuncDecl> node) {
        std::cout << "FUNC\n";
        for(auto& stmt: node->body) {
            if (stmt->is_def()) {
                stmt->as_def()->accept(this);
            }
        }
    }

    void JlFunctionInitializer::visit(std::shared_ptr<Module> node) {
        std::cout << "MODULE\n";
        for(auto& comp: node->decls) {
            if (comp->is_def()) {
                comp->as_def()->accept(this);
            }

            if (comp->is_decl() && !comp->is_builtin()) {
                comp->as_decl()->accept(this);
            }
        }
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
            *oss << node->indices[i]->dump();
            if (i != node->indices.size() - 1) {
                *oss << ",";
            }
        }
        *oss << "]";
    }

    void FinchCompileVisitor::visit(std::shared_ptr<Definition> node) {
        auto old_oss = oss;
        for (auto& acc: node->lhs) {
            std::stringstream ss;
            oss = &ss;
            ss << "ctx = Finch.LowerJulia()\ncode = Finch.contain(ctx) do ctx_2\n";
            auto v = DefinitionVisitor(&ss);
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
            for (size_t i=0; i < node->reduction_list.size(); i++) {
                auto& red = node->reduction_list[i];
                ss << "w_" << i << "[] where (@loop " << red->reductionVar->name << " w_" << i << "[] ";
                red->reductionOp->accept(this);
                *oss << "= ";
            }

            node->rhs->accept(this);

            for (size_t i=0; i < node->reduction_list.size(); i++) {
                ss << ")";
            }
            for (auto& idx: acc->indices) {
                ss << ")";
            }
            ss << "\n";
            ss << "kernel_code = Finch.execute_code_virtualized(kernel, ctx_2)\n";
            ss << "end\n";
            ss << "return quote\n";
            ss << "     function def_" << def_id << "(";
            for (auto curr = v.tensors.begin();curr != v.tensors.end();) {
                ss << (*curr);
                if (++curr != v.tensors.end()) {
                    ss << ", ";
                }
            }
            ss << ")\n";
            ss << "         $code\n";
            ss << "     end\n";
            ss << "end\n";
            std::string virtualized =ss.str();
            const char* vc = virtualized.data();
//            std::cout << vc;
            jl_value_t* expr = finch_exec(vc);
            jl_value_t* code = finch_exec("repr(last(%s.args))", expr);
            auto s = jl_string_data(code);

            *old_oss << "finch_def_code_" << def_id << " = finch_eval(\n";

            std::stringstream ss_(s);
            std::string line;
            while(std::getline(ss_, line, '\n')){
                *old_oss << "\"" << line << "\"\n";
            }
            *old_oss << ");\n";
            def_id += 1;
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

    void FinchCompileVisitor::visit(std::shared_ptr<Module> node) {
        for(auto& decl: node->decls) {
            if (decl->is_decl() && !decl->is_builtin()) {
                decl->as_decl()->accept(this);
            }

            if (decl->is_def()) {
                decl->as_def()->accept(this);
            }
        }
    }

    void FinchCompileVisitor::visit(std::shared_ptr<FuncDecl> node) {
        for(auto& stmt: node->body) {
            if (stmt->is_def()) {
                stmt->as_def()->accept(this);
            }
        }
    }

}