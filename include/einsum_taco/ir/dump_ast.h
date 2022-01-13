//
// Created by Alexandra Dima on 11.01.2022.
//

#ifndef EINSUM_TACO_DUMP_AST_H
#define EINSUM_TACO_DUMP_AST_H

#include<einsum_taco/ir/ir.h>

namespace einsum{
    class DumpAstVisitor : public IRVisitor {
    public:
        std::string ast;
        int indent_;

        void visit(const IndexVar& node) override;

        void visit(const Literal& node) override;
        void visit(const TensorVar& node) override;
        void visit(const IndexVarExpr& node) override;
        void visit(const Access& node) override;
        void visit(const ReadAccess& node) override;
        void visit(const BinaryOp& node) override;
        void visit(const UnaryOp& node) override;
        void visit(const Definition& node) override;
        void visit(const FuncDecl& node) override;
        void visit(const Call& node) override;
        void visit(const CallStarRepeat& node) override;
        void visit(const CallStarCondition& node) override;
        void visit(const Module& node) override;
        void visit(const Reduction& node) override;

        std::string get_indent() const;

        void indent();

        void unindent();

        template<typename T>
        void visit_access(const T& node);

        template<typename T>
        std::vector<std::string> visit_array(std::vector<T> arr);

        void array_ast(const std::vector<std::string>& arr);
    };
}

#endif //EINSUM_TACO_DUMP_AST_H
