//
// Created by Alexandra Dima on 11.01.2022.
//

#include <einsum_taco/ir/dump_ast.h>
#include <string>

using namespace std;

void einsum::DumpAstVisitor::indent()  {
    indent_ += 1;
}

std::string einsum::DumpAstVisitor::get_indent() const {
    return std::string(indent_, '\t');
}

void einsum::DumpAstVisitor::unindent() {
    indent_ -= 1;
}

void einsum::DumpAstVisitor::visit(const einsum::IndexVar &node) {
    indent();
    node.dimension->accept(this);
    auto dim = ast;
    ast = get_indent() + "<" + node.class_name() + "\n";
    indent();
    ast += get_indent() + node.name + "\n";
    unindent();
    ast += dim + "\n";
    ast += get_indent() + ">";
    unindent();
}

void einsum::DumpAstVisitor::visit(const einsum::Literal &node) {
    indent();
    ast = get_indent() + "<" + node.class_name() + " " + node.dump() + " " + node.getDatatype()->dump() + ">";
    unindent();
}

void einsum::DumpAstVisitor::visit(const einsum::Reduction &node) {
    indent();

    node.reductionVar->accept(this);
    auto red_var = ast;

    node.reductionInit->accept(this);
    auto red_init = ast;

    ast = get_indent() + "<" + node.class_name() + "\n" +
            red_var + "\n";
    indent();
    ast += get_indent() + node.reductionOp->reductionSign + "\n";
    unindent();
    ast += red_init + "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit_binary(const einsum::BinaryOp &node) {
    indent();

    node.left->accept(this);
    auto left = ast;

    node.right->accept(this);
    auto right = ast;

    ast = get_indent() + "<" + node.class_name() + "\n" +
            left + "\n";
    indent();
    ast += get_indent() + node.op->sign + "\n";
    unindent();
    ast += right + "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit_unary(const einsum::UnaryOp &node) {
    indent();

    node.expr->accept(this);
    auto exp = ast;

    ast = get_indent() + "<" + node.class_name() + "\n";
    indent();
    ast += get_indent() + node.op->sign + "\n";
    unindent();
    ast += exp + "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(const einsum::ModuloExpression &node) {
    visit_binary(node);
}

void einsum::DumpAstVisitor::visit(const einsum::ArithmeticExpression &node) {
    visit_binary(node);
}

void einsum::DumpAstVisitor::visit(const einsum::LogicalExpression &node) {
    visit_binary(node);
}

void einsum::DumpAstVisitor::visit(const einsum::ComparisonExpression &node) {
    visit_binary(node);
}

void einsum::DumpAstVisitor::visit(const einsum::NotExpression &node) {
    visit_unary(node);
}

void einsum::DumpAstVisitor::visit(const einsum::TensorVar &node) {
    indent();
    ast = get_indent() + "<" + node.class_name() + " " + node.name + " " + node.getType()->dump() + ">";
    unindent();
}

void einsum::DumpAstVisitor::visit(const einsum::IndexVarExpr &node) {
    indent();
    node.indexVar->accept(this);
    auto ivar = ast;

    ast = get_indent() + "<" + node.class_name() + "\n" +
            ivar + "\n" +
            get_indent() + ">";

    unindent();
}

template<typename T>
void einsum::DumpAstVisitor::visit_access(const T& node) {
    indent();
    node.tensor->accept(this);
    auto tensor = ast;

    auto indice_asts = visit_array(node.indices);

    ast = get_indent() + "<" + node.class_name() + "\n" +
          tensor + "\n";

    array_ast(indice_asts);

    ast += "\n" + get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(const einsum::Access &node) {
    visit_access(node);
}

void einsum::DumpAstVisitor::visit(const einsum::ReadAccess &node) {
    visit_access(node);
}

void einsum::DumpAstVisitor::visit(const einsum::Definition &node) {
    indent();

    auto lhs = visit_array(node.lhs);
    node.rhs->accept(this);
    auto rhs = ast;
    auto reds = visit_array(node.reductionVars);

    ast = get_indent() + "<" + node.class_name() + "\n";
    array_ast(lhs);
    ast += "\n";
    ast += rhs + "\n";
    array_ast(reds);
    ast += "\n" + get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(const einsum::FuncDecl &node) {
    indent();

    auto inputs = visit_array(node.inputs);
    auto outputs = visit_array(node.outputs);
    auto body = visit_array(node.body);

    ast = get_indent() + "<" + node.class_name() + "\n";
    indent();
    ast += get_indent() + node.funcName + "\n";
    unindent();
    array_ast(inputs);
    ast += "\n";
    array_ast(outputs);
    ast += "\n";
    array_ast(body);
    ast += "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(const einsum::Call &node) {
    indent();
    auto args = visit_array(node.arguments);

    node.function->accept(this);
    auto func = ast;

    ast = get_indent() + "<" + node.class_name() + "\n" +
            func + "\n";
    array_ast(args);
    ast += "\n" + get_indent() + ">";

    unindent();

}

void einsum::DumpAstVisitor::visit(const einsum::CallStarRepeat &node) {
    indent();
    auto args = visit_array(node.arguments);

    node.function->accept(this);
    auto func = ast;

    ast = get_indent() + "<" + node.class_name() + "\n" +
          func + "\n";
    array_ast(args);
    ast += "\n";
    indent();
    ast += get_indent() + std::to_string(node.numIterations) + "\n";
    unindent();
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(const einsum::CallStarCondition &node) {
    indent();
    auto args = visit_array(node.arguments);

    node.function->accept(this);
    auto func = ast;

    node.stopCondition->accept(this);
    auto stop = ast;

    ast = get_indent() + "<" + node.class_name() + "\n" +
          func + "\n";
    array_ast(args);
    ast += "\n" + stop + "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(const einsum::Module &node) {
    auto components = visit_array(node.decls);

    ast = "<" + node.class_name() + "\n";
    array_ast(components);
    ast += "\n>";

}

template<typename T>
vector<string> einsum::DumpAstVisitor::visit_array(vector<T> arr) {
    indent();
    indent();

    auto strings = vector<string>();

    for(auto &&a: arr) {
        a->accept(this);
        strings.push_back(ast);
    }

    unindent();
    unindent();

    return strings;
}

void einsum::DumpAstVisitor::array_ast(const std::vector<std::string>& arr) {
    indent();

    if (arr.empty()) {
        ast += get_indent() + "<Array>";
        unindent();
        return;
    }

    ast += get_indent() + "<Array" + "\n";
    for (auto &&a: arr) {
        ast += a + "\n";
    }
    ast += get_indent() + ">";

    unindent();
}

