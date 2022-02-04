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
        int indent_ = 0;

        void visit(std::shared_ptr<IndexVar> node) override;

        void visit(std::shared_ptr<Literal> node) override;
        void visit(std::shared_ptr<TensorVar> node) override;
        void visit(std::shared_ptr<IndexVarExpr> node) override;
        void visit(std::shared_ptr<Access> node) override;
        void visit(std::shared_ptr<ReadAccess> node) override;
        void visit(std::shared_ptr<BinaryOp> node) override;
        void visit(std::shared_ptr<UnaryOp> node) override;
        void visit(std::shared_ptr<Definition> node) override;
        void visit(std::shared_ptr<FuncDecl> node) override;
        void visit(std::shared_ptr<Call> node) override;
        void visit(std::shared_ptr<CallStarRepeat> node) override;
        void visit(std::shared_ptr<CallStarCondition> node) override;
        void visit(std::shared_ptr<Module> node) override;
        void visit(std::shared_ptr<Reduction> node) override;
        void visit(std::shared_ptr<Datatype> node) override;
        void visit(std::shared_ptr<TensorType> node) override;
        void visit(std::shared_ptr<TupleType> node) override;
        void visit(std::shared_ptr<Operator> node) override;

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
