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
    einsum_iassert(indent_ >= 0);
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<IndexVar> node) {
    indent();
    auto dim = ast;
    ast = get_indent() + "<" + node->class_name() + "\n";
    indent();
    ast += get_indent() + node->name + "\n";
    unindent();
    ast += dim + "\n";
    ast += get_indent() + ">";
    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<Literal> node) {
    indent();
    ast = get_indent() + "<" + node->class_name() + " " + node->dump() + " " + node->getDatatype()->dump() + ">";
    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<Reduction> node) {
    indent();

    node->reductionVar->accept(this);
    auto red_var = ast;

    node->reductionInit->accept(this);
    auto red_init = ast;

    ast = get_indent() + "<" + node->class_name() + "\n" +
            red_var + "\n";
    indent();
    ast += get_indent() + node->reductionOp->op->reductionSign + "\n";
    unindent();
    ast += red_init + "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<TensorVar> node) {
    indent();

    node->getType()->accept(this);
    auto type = ast;

    ast = get_indent() + "<" + node->class_name() + " " + node->name + "\n";
    ast += type + "\n";
    ast += get_indent() + "isGlobal " + std::to_string(node->is_global) + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<IndexVarExpr> node) {
    indent();
    node->indexVar->accept(this);
    auto ivar = ast;

    ast = get_indent() + "<" + node->class_name() + "\n" +
            ivar + "\n" +
            get_indent() + ">";

    unindent();
}

template<typename T>
void einsum::DumpAstVisitor::visit_access(const T& node) {
    indent();
    node->tensor->accept(this);
    auto tensor = ast;

    auto indice_asts = visit_array(node->indices);

    ast = get_indent() + "<" + node->class_name() + "\n" +
          tensor + "\n";

    array_ast(indice_asts);

    ast += "\n" + get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<Access> node) {
    visit_access(node);
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<ReadAccess> node) {
    visit_access(node);
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<TupleVar> node) {
    ast += node->name;
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<TupleVarReadAccess> node) {
    ast += "<" + std::to_string(node->idx) + ">(";
    node->var->accept(this);
    ast += ")";
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<MultipleOutputDefinition> node) {
    node->lhs->accept(this);
    ast += " = ";
    node->rhs->accept(this);
    ast += "\n";
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<Definition> node) {
    std::cout << node->dump() << "\n";
    indent();

    auto lhs = visit_array(node->lhs);
    node->rhs->accept(this);
    auto rhs = ast;
    auto reds = visit_array(node->reduction_list);

    ast = get_indent() + "<" + node->class_name() + "\n";
    array_ast(lhs);
    ast += "\n";
    ast += rhs + "\n";
    array_ast(reds);
    ast += "\n" + get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<MemAssignment> node) {
    indent();

    node->rhs->accept(this);
    auto rhs = ast;

    node->lhs->accept(this);
    auto lhs = ast;

    ast = get_indent() + "<" + node->class_name() + "\n";
    ast += rhs + "\n";
    ast += lhs + "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<Allocate> node) {
    indent();

    node->tensor->accept(this);
    auto t = ast;

    ast = get_indent() + "<" + node->class_name() + "\n";
    ast += t;
    ast += "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<FuncDecl> node) {
    indent();

    auto inputs = visit_array(node->inputs);
    auto outputs = visit_array(node->outputs);
    auto body = visit_array(node->body);

    ast = get_indent() + "<" + node->class_name() + "\n";
    indent();
    ast += get_indent() + node->funcName + "\n";
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

void einsum::DumpAstVisitor::visit(std::shared_ptr<Call> node) {
    indent();
    auto args = visit_array(node->arguments);

    node->function->accept(this);
    auto func = ast;

    ast = get_indent() + "<" + node->class_name() + "\n" +
            func + "\n";
    array_ast(args);
    ast += "\n" + get_indent() + ">";

    unindent();

}

void einsum::DumpAstVisitor::visit(std::shared_ptr<CallStarRepeat> node) {
    indent();
    auto args = visit_array(node->arguments);

    node->function->accept(this);
    auto func = ast;

    ast = get_indent() + "<" + node->class_name() + "\n" +
          func + "\n";
    array_ast(args);
    ast += "\n";
    indent();
    ast += get_indent() + std::to_string(node->numIterations) + "\n";
    unindent();
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<CallStarCondition> node) {
    indent();
    auto args = visit_array(node->arguments);

    node->function->accept(this);
    auto func = ast;

    node->stopCondition->accept(this);
    auto stop = ast;

    ast = get_indent() + "<" + node->class_name() + "\n" +
          func + "\n";
    array_ast(args);
    ast += "\n" + stop + "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<Module> node) {
    auto components = visit_array(node->decls);

    ast = "<" + node->class_name() + "\n";
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

void einsum::DumpAstVisitor::visit(std::shared_ptr<BinaryOp> node) {
    indent();

    node->left->accept(this);
    auto left = ast;

    node->right->accept(this);
    auto right = ast;

    node->op->accept(this);
    auto op = ast;

    ast = get_indent() + "<" + node->class_name() + "\n";
    ast += left + "\n";
    ast += op + "\n";
    ast += right + "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(std::shared_ptr<UnaryOp> node) {
    indent();

    node->expr->accept(this);
    auto exp = ast;
    node->op->accept(this);
    auto op = ast;

    ast = get_indent() + "<" + node->class_name() + "\n";
    ast += op + "\n";
    ast += exp + "\n";
    ast += get_indent() + ">";

    unindent();
}

void einsum::DumpAstVisitor::visit(shared_ptr<Datatype> node) {
    ast = get_indent() + "<" + node->class_name() + "\n";
    indent();
    ast += get_indent() + node->dump() + "\n";
    unindent();
    ast += get_indent() + ">";
}

void einsum::DumpAstVisitor::visit(shared_ptr<TensorType> node) {

    node->getElementType()->accept(this);
    auto type = ast;

    auto dimensions = visit_array(node->getDimensions());

    ast = get_indent() + "<" + node->class_name() + "\n";
    ast += type + "\n";
    array_ast(dimensions);
    ast += get_indent() + ">";
}

void einsum::DumpAstVisitor::visit(shared_ptr<TupleType> node) {
    auto types = visit_array(node->tuple);

    ast = get_indent() + "<" + node->class_name() + "\n";
    array_ast(types);
    ast += get_indent() + ">";

}

void einsum::DumpAstVisitor::visit(shared_ptr<Operator> node) {
    ast = get_indent() + "<" + node->class_name() + "\n";
    indent();
    ast += get_indent() + node->sign + "\n";
    unindent();
    ast += get_indent() + ">";
}

void einsum::DumpAstVisitor::visit(shared_ptr<Initialize> node) {
    node->tensor->accept(this);
    auto t = ast;

    ast = get_indent() + "<" + node->class_name() + "\n";
    ast += t + "\n";
    ast += get_indent() + ">";
}

void einsum::DumpAstVisitor::visit(shared_ptr<AndOperator> node) {

}

void einsum::DumpAstVisitor::visit(shared_ptr<OrOperator> node) {

}

void einsum::DumpAstVisitor::visit(shared_ptr<AddOperator> node) {

}

void einsum::DumpAstVisitor::visit(shared_ptr<MulOperator> node) {

}

void einsum::DumpAstVisitor::visit(shared_ptr<MinOperator> node) {

}

void einsum::DumpAstVisitor::visit(shared_ptr<ChooseOperator> node) {

}

std::string einsum::DumpAstVisitor::name() {
    return "DumpAstVisitor";
}


