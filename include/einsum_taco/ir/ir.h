//
// Created by Alexandra Dima on 10/13/21.
//

#ifndef EINSUM_TACO_IR_H
#define EINSUM_TACO_IR_H

#include<string>
#include<vector>
#include<einsum_taco/type.h>


namespace einsum {
    enum class IRNodeType {
        FuncDecl,
        ParamDecl,
        Definition,
        TensorDefinition
    };

    class IRVisitor;

    template<typename T, IRNodeType type>
    struct IRNode {

        void accept(IRVisitor *v) const;

        IRNodeType getType() const {
            return type;
        }
    };

    struct ParamDeclNode : IRNode<ParamDeclNode, IRNodeType::ParamDecl> {
        std::string paramName;
        einsum::Type paramType;

        ParamDeclNode(std::string paramName, einsum::Type paramType) : paramName(paramName), paramType(paramType) {}
    };

    struct DefinitionNode : IRNode<DefinitionNode, IRNodeType::Definition> {

    };

    struct FuncDeclNode : IRNode<FuncDeclNode, IRNodeType::FuncDecl> {
        std::string funcName;
        std::vector<ParamDeclNode> inParams;
        std::vector<ParamDeclNode> outParams;
        std::vector<DefinitionNode> body;

        FuncDeclNode(std::string funcName, std::vector<ParamDeclNode> inParams, std::vector<ParamDeclNode> outParams, std::vector<DefinitionNode> body)
            : funcName(funcName), inParams(inParams), outParams(outParams), body(body) {}
    };

    struct IRVisitor {
        virtual void accept(const FuncDeclNode& node) = 0;
        virtual void accept(const ParamDeclNode& node) = 0;
        virtual void accept(const DefinitionNode& node) = 0;
    };

    template<typename T, IRNodeType type>
    void IRNode<T, type>::accept(IRVisitor *v) const {
        v->accept(static_cast<const T&>(*this));
    }
}


#endif //EINSUM_TACO_IR_H
