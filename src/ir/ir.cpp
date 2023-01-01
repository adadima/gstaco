//
// Created by Alexandra Dima on 10/13/21.
//
#include <einsum_taco/ir/type.h>
#include <einsum_taco/ir/ir.h>
#include <string>
#include <iostream>
#include <einsum_taco/ir/context.h>
#include <set>
#include <map>
#include <cxxabi.h>
#include <algorithm>
#include <limits.h>


namespace einsum {

    std::string IR::class_name() const {
        int status;
        char* demangled_name_buf = abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, &status);
        auto demangled_name = std::string{demangled_name_buf};
        free(demangled_name_buf);
        if (demangled_name.find("einsum::") == 0) {
            return demangled_name.substr(8);
        }
        return demangled_name;
    }


    bool Literal::isInt() const {
        return this->getDatatype()->isInt();
    }

    bool Literal::isFloat() const {
        return this->getDatatype()->isFloat();
    }

    bool Literal::isBool() const {
        return this->getDatatype()->isBool();
    }

    std::shared_ptr<Type> Literal::getType() const {
        return type;
    }

    std::string Literal::dump() const {
        switch (this->getDatatype()->getKind()) {
            case Datatype::Kind::Int:
                return std::to_string(this->getValue<int>());
            case Datatype::Kind::Float:
                return std::to_string(this->getValue<float>());
            case Datatype::Kind::Bool:
                return this->getValue<bool>() ? "true" : "false";
//                return std::to_string(this->getValue<bool>());
        }
    }

    std::vector<std::shared_ptr<IndexVar>> Literal::getIndices() {
        return {};
    }

    std::string BinaryOp::dump() const {
        auto left_ = this->left->dump();
        auto right_ = this->right->dump();
        if (this->left->precedence > this->precedence) {
            left_ = "(" + left_ + ")";
        }
        if ((this->right->precedence > this->precedence) ||  (this->right->precedence == this->precedence && this->isAsymmetric)){
            right_ = "(" + right_ + ")";
        }
        return left_ + " " + this->op->sign + " " + right_;
    }

//    std::string BinaryOp::fdump() const {
//        auto left_ = this->left->fdump();
//        auto right_ = this->right->fdump();
//        if (this->left->precedence > this->precedence) {
//            left_ = "(" + left_ + ")";
//        }
//        if ((this->right->precedence > this->precedence) ||  (this->right->precedence == this->precedence && this->isAsymmetric)){
//            right_ = "(" + right_ + ")";
//        }
//        return left_ + " " + this->op->sign + " " + right_;
//    }

    std::vector<std::shared_ptr<IndexVar>> BinaryOp::getIndices() {
        std::vector<std::shared_ptr<IndexVar>> lIndices = this->left->getIndices();
        std::vector<std::shared_ptr<IndexVar>> rIndices = this->right->getIndices();
        std::vector<std::shared_ptr<IndexVar>> inds(lIndices);
        inds.insert(inds.end(), lIndices.begin(), lIndices.end());
        inds.insert(inds.end(), rIndices.begin(), rIndices.end());
        return inds;
    }

    std::string UnaryOp::dump() const {
        return this->op->sign + " " + this->expr->dump();
    }

    std::vector<std::shared_ptr<IndexVar>> UnaryOp::getIndices() {
        return this->expr->getIndices();
    }

    std::string TensorVar::dump() const {
        return this->name;
    }

    std::shared_ptr<TensorType> TensorVar::getType() const {
        return this->type;
    }

    bool TensorVar::is_global_var() const {
        return is_global;
    }

    std::string IndexVar::dump() const {
        return this->name;
    }

    std::string IndexVarExpr::dump() const {
        return this->indexVar->name;
    }

    std::vector<std::shared_ptr<IndexVar>> IndexVarExpr::getIndices() {
        return {this->indexVar};
    }

    std::shared_ptr<Type> IndexVarExpr::getType() const {
        return std::make_shared<Datatype>(Datatype::Kind::Int);
    }

    std::string Access::dump() const {
        auto acc = this->tensor->name;
        for (const auto &indice : this->indices) {
            acc += "[" + indice->dump() + "]";
        }
        return acc;
    }

    std::string ReadAccess::dump() const {
        auto acc = this->tensor->name;
        for (const auto &indice : this->indices) {
            acc += "[" + indice->dump() + "]";
        }
        return acc;
    }

//    std::string ReadAccess::fdump() const {
//        if (tensor->getType() == 0) {
//            return "$" + tensor->name;
//        }
//        auto acc = this->tensor->name + "[";
//        for (size_t i=0; i < indices.size(); i++) {
//            acc += indices[i]->dump();
//            if (i != indices.size() - 1) {
//                acc += ",";
//            }
//        }
//        acc += "]";
//        return acc;
//    }

    std::vector<std::shared_ptr<IndexVar>> ReadAccess::getIndices() {
        std::vector<std::shared_ptr<IndexVar>> indices;
        for (const auto &indExpr : this->indices) {
            auto inds = indExpr->getIndices();
            indices.insert(indices.end(), inds.begin(), inds.end());
        }
        return indices;
    }

    std::shared_ptr<Type> ReadAccess::getType() const {
        std::vector<std::shared_ptr<Expression>> dims;
        int last = this->tensor->getOrder() - this->indices.size();
        if (last < 0) {
            last = 0;
        }
        dims.reserve(last);
        for (int i=0; i < last; i++) {
            dims.push_back(this->tensor->getDimensions()[i]);
        }
        if (dims.empty()) {
            return this->tensor->getType()->getElementType();
        }
        return std::make_shared<TensorType>(this->tensor->getType()->getElementType(), dims);
    }

    std::string Reduction::dump() const {
        return this->reductionVar->dump() + ":" + "(" + this->reductionOp->op->reductionSign + ", " + this->reductionInit->dump() + ")";
    }

    std::shared_ptr<Reduction> Reduction::orReduction(std::shared_ptr<IndexVar> var) {
        return IR::make<Reduction>(var, or_red, IR::make<Literal>(0, IR::make<Datatype>(Datatype::Kind::Int)));
    }

    std::shared_ptr<Reduction> Reduction::andReduction(std::shared_ptr<IndexVar> var) {
        return IR::make<Reduction>(var, IR::make<AndOperator>(), IR::make<Literal>(1, IR::make<Datatype>(Datatype::Kind::Int)));
    }

    std::shared_ptr<Reduction> Reduction::addReduction(std::shared_ptr<IndexVar> var) {
        return IR::make<Reduction>(var, IR::make<AddOperator>(), IR::make<Literal>(0, IR::make<Datatype>(Datatype::Kind::Int)));
    }

    std::shared_ptr<Reduction> Reduction::minReduction(std::shared_ptr<IndexVar> var) {
        return IR::make<Reduction>(var, IR::make<MinOperator>(), IR::make<Literal>(std::numeric_limits<int>::max(), IR::make<Datatype>(Datatype::Kind::Int)));
    }

    std::shared_ptr<Reduction> Reduction::chooseReduction(std::shared_ptr<IndexVar> var) {
        return IR::make<Reduction>(var, IR::make<ChooseOperator>(), IR::make<Literal>(0, IR::make<Datatype>(Datatype::Kind::Int)));
    }

    std::string Definition::dump() const {
        std::string def; // = this->lhs->dump() + " = " + this->rhs->dump();
        bool first = true;
        for (const auto & lh : this->lhs) {
            if (!first) {
                def += ", ";
            } else {
                first = false;
            }
            def += lh->dump();
        }

        def += " = " + this->rhs->dump();
        if (this->reduction_list.empty()) {
            return def;
        }
        def += " | ";
        first = true;
        for (auto& red: this->reduction_list) {
            if (!first) {
                def += ", ";
            } else {
                first = false;
            }
            def += red->dump();
        }
        return def;
    }

    std::string FuncDecl::dump() const {
        std::string inParams;
        auto inputs = this->inputs;
        auto outputs = this->outputs;
        auto inputTypes = this->getInputType();
        auto outputTypes = this->getOutputType();

        for (int i=0; i < inputs.size(); i++) {
            inParams += (inputs[i]->dump() + " " + inputTypes[i]->dump());
            if (i < inputs.size() - 1) {
                inParams += ", ";
            }
        }

        std::string outParams;
        for (int i=0; i < outputs.size(); i++) {
            outParams += (outputs[i]->dump() + " " + outputs[i]->getType()->dump());
            if (i < outputs.size() - 1) {
                outParams += ", ";
            }
        }

        auto def = "Let " + this->funcName + "(" + inParams + ") -> (" + outParams + ")\n";

        std::string body;
        for (const auto &i : this->body) {
            body += "    " + i->dump();
            body += "\n";
        }
        return def + body + "End";
    }

    std::shared_ptr<Type> Call::getType() const {
        return this->function->getOutputType();
    }

    std::string Call::dump_args() const {
        std::string args;
        for (int i=0; i < this->arguments.size(); i++) {
            args += this->arguments[i]->dump();
            if (i < this->arguments.size() - 1) {
                args += ", ";
            }
        }
        return "(" + args + ")";
    }

    std::string Call::dump() const {
        return this->function->funcName + this->dump_args();
    }

    std::vector<std::shared_ptr<IndexVar>> Call::getIndices() {
        std::vector<std::shared_ptr<IndexVar>> inds;
        for (auto &argument : this->arguments) {
            auto newInds = argument->getIndices();
            inds.insert(inds.end(), newInds.begin(), newInds.end());
        }
        return inds;
    }

    std::string CallStarRepeat::dump() const {
        auto call = this->function->funcName + "*" + this->dump_args();
        std::string rules;
        for (auto& rule: format_rules) {
            rules += rule->dump() + "   ,   ";
        }
        return call + " | " + std::to_string(this->numIterations) + rules;
    }

    std::string CallStarCondition::dump() const {
        auto call = this->function->funcName + "*" + this->dump_args();
        std::string rules;
        for (auto& rule: format_rules) {
            rules += rule->dump() + "   ,   ";
        }
        //  TODO: think of when to wrap condition around parens
        return call + " | " + "(" + this->stopCondition->dump() + ")" + rules;
    }

    std::vector<std::shared_ptr<Type>> FuncDecl::getInputType() const {
        std::vector<std::shared_ptr<Type>> types;
        auto mapper = [](const std::shared_ptr<TensorVar> tvar) { return tvar->getType(); };
        std::transform(this->inputs.begin(), this->inputs.end(), std::back_inserter(types), mapper);
        return types;
    }

    //TODO:: make this return vector of types
    std::shared_ptr<TupleType> FuncDecl::getOutputType() const {
        std::vector<std::shared_ptr<Type>> types;
        for (auto &var : outputs) {
            types.push_back(var->getType());
        }
        return Type::make<TupleType>(types);
    }

    std::string Module::dump() const {
        std::string code;
        for (const auto &d: decls) {
            code += "\n";
            code += d->dump();
            if (d->is_var()) {
                code += " " + d->as_var()->getType()->dump();
            }
            code += "\n";
        }
        return code;
    }

    void Module::add(std::shared_ptr<ModuleComponent> decl) {
        decls.push_back(std::move(decl));
    }

    std::vector<std::shared_ptr<TensorVar>> Module::get_globals() const {
        auto globals = std::vector<std::shared_ptr<TensorVar>>();
        for (auto &comp : decls) {
            if (comp->is_var()) {
                globals.push_back(comp->as_var());
            }
            if (comp->is_init()) {
                globals.push_back(comp->as_init()->tensor);
            }
            if (comp->is_def()) {
                auto def = comp->as_def();
                auto lhs = std::vector<std::shared_ptr<TensorVar>>();
                for(auto &acc: def->lhs) {
                    lhs.push_back(acc->tensor);
                }
                globals.insert(globals.end(), lhs.begin(), lhs.end());
            }
        }
        return globals;
    }

    bool ModuleComponent::is_decl() const {
        return dynamic_cast<const FuncDecl*>(this) != nullptr;
    }

    std::shared_ptr<FuncDecl> ModuleComponent::as_decl() {
        try {
            return std::dynamic_pointer_cast<FuncDecl>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_def() const {
        return dynamic_cast<const Definition*>(this) != nullptr;
    }

    std::shared_ptr<Definition> ModuleComponent::as_def() {
        try {
            return std::dynamic_pointer_cast<Definition>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_multi_def() const {
        return dynamic_cast<const MultipleOutputDefinition*>(this) != nullptr;
    }

    std::shared_ptr<MultipleOutputDefinition> ModuleComponent::as_multi_def() {
        try {
            return std::dynamic_pointer_cast<MultipleOutputDefinition>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_expr() const {
        return dynamic_cast<const Expression*>(this) != nullptr;
    }

    std::shared_ptr<Expression> ModuleComponent::as_expr() {
        try {
            return std::dynamic_pointer_cast<Expression>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_var() const {
        return dynamic_cast<const TensorVar*>(this) != nullptr;
    }

    std::shared_ptr<TensorVar> ModuleComponent::as_var() {
        try {
            return std::dynamic_pointer_cast<TensorVar>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_allocate() const {
        return dynamic_cast<const Allocate*>(this) != nullptr;
    }

    std::shared_ptr<Allocate> ModuleComponent::as_allocate() {
        try {
            return std::dynamic_pointer_cast<Allocate>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_mem_assign() const {
        return dynamic_cast<const MemAssignment*>(this) != nullptr;
    }

    std::shared_ptr<MemAssignment> ModuleComponent::as_mem_assign() {
        try {
            return std::dynamic_pointer_cast<MemAssignment>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_init() const {
        return dynamic_cast<const Initialize*>(this) != nullptr;
    }

    bool ModuleComponent::is_tuple_var() const {
        return dynamic_cast<const TupleVar*>(this) != nullptr;
    }

    bool ModuleComponent::is_format_rule() const {
        return dynamic_cast<const FormatRule*>(this) != nullptr;
    }

    std::shared_ptr<FormatRule> ModuleComponent::as_format_rule() {
        try {
            return std::dynamic_pointer_cast<FormatRule>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    std::shared_ptr<Initialize> ModuleComponent::as_init() {
        try {
            return std::dynamic_pointer_cast<Initialize>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    bool ModuleComponent::is_builtin() const {
        return dynamic_cast<const BuiltinFuncDecl*>(this) != nullptr;
    }

    std::shared_ptr<BuiltinFuncDecl> ModuleComponent::as_builtin() {
        try {
            return std::dynamic_pointer_cast<BuiltinFuncDecl>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    std::shared_ptr<TupleVar> ModuleComponent::as_tuple_var() {
        try {
            return std::dynamic_pointer_cast<TupleVar>(this->shared_from_this());
        } catch (const std::bad_weak_ptr& exp) {
            std::abort();
        }
    }

    std::string IndexVar::getName() const {
        return name;
    }

    std::string IndexVarExpr::getName() const {
        return indexVar->getName();
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  Literal::getIndexVarDims(IRContext* context) const {
        return {};
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  BinaryOp::getIndexVarDims(IRContext* context) const {
        auto left_dims = left->getIndexVarDims(context);
        auto right_dims = right->getIndexVarDims(context);

        for (const auto &[key, value] : right_dims) {
            left_dims.emplace(key, value);
        }
        return left_dims;
    }

    std::shared_ptr<Type> BinaryOp::getType() const {
        return type;
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  UnaryOp::getIndexVarDims(IRContext* context) const {
        return expr->getIndexVarDims(context);
    }

    std::shared_ptr<Type> UnaryOp::getType() const {
        return type;
    }

    template<class T, class E>
    std::map<std::string, std::set<std::shared_ptr<Expression>>>  getDimsFromAccess(const std::shared_ptr<const TensorVar>& tensor, const std::vector<std::shared_ptr<E>>& indices) {
        auto dims = std::map<std::string, std::set<std::shared_ptr<Expression>>> ();
        std::cout << "Tensor: " << tensor->name << ", " << tensor->type->dump() << "\n";
        std::cout << "Tensor indices: " << indices.size() << "\n";
        for (int i=0; i < indices.size(); i++) {
            auto ind = indices[i];
            auto index_var = std::dynamic_pointer_cast<T>(ind);
            if (index_var) {
                auto dimension = tensor->getDimensions()[i];
                dims[index_var->getName()].insert(dimension);
            }
        }

        return dims;
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  ReadAccess::getIndexVarDims(IRContext* context) const {
        return getDimsFromAccess<IndexVarExpr, Expression>(context->get_read_tensor(tensor), indices);
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  IndexVarExpr::getIndexVarDims(IRContext* context) const {
        return {};
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  Call::getIndexVarDims(IRContext* context) const {
        auto dims = std::map<std::string, std::set<std::shared_ptr<Expression>>> ();

        for (auto &&arg : arguments) {
            for (const auto &[key, value] : arg->getIndexVarDims(context)) {
                auto &entry = dims[key];
                entry.insert(value.begin(), value.end());
            }
        }

        return dims;
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>>  Access::getIndexVarDims(IRContext* context) const {
        auto dims = std::map<std::string, std::set<std::shared_ptr<Expression>>> ();

        for (auto &idx: indices) {
            for (const auto &[key, value] : idx->getIndexVarDims(context)) {
                auto &entry = dims[key];
                entry.insert(value.begin(), value.end());
            }
        }

        return dims;
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>> Definition::getIndexVarDims(IRContext* context) const {
        auto dims = std::map<std::string, std::set<std::shared_ptr<Expression>>> ();

        for (auto &&acc : lhs) {
            for (const auto &[key, value] : acc->getIndexVarDims(context)) {
                auto &entry = dims[key];
                entry.insert(value.begin(), value.end());
            }
        }

        for (const auto &[key, value] : rhs->getIndexVarDims(context)) {
            auto &entry = dims[key];
            entry.insert(value.begin(), value.end());
        }

        return dims;
    }

    std::set<std::string> Definition::getLeftIndexVars(IRContext* context) const {
        auto vars = std::set<std::string>();

        for (auto &&acc : lhs) {
            for (const auto &[key, value] : acc->getIndexVarDims(context)) {
                vars.insert(key);
            }
        }

        for (const auto &[key, value] : rhs->getIndexVarDims(context)) {
            vars.insert(key);
        }

        return vars;
    }

    std::set<std::string> Definition::getReductionVars() const {
        auto s = std::set<std::string>();
        for (auto &&var: reduction_list) {
            s.insert(var->reductionVar->getName());
        }
        return s;
    }

    std::string Allocate::dump() const {
        return "malloc " + tensor->name + "\n";
    }

    std::string MemAssignment::dump() const {
        return lhs->dump() + " = " + rhs->dump() + "\n";
    }

    std::string Initialize::dump() const {
        return tensor->type->dump() + " " + tensor->name;
    }

    void DefaultIRVisitor::visit(std::shared_ptr<IndexVar> node) { throw std::runtime_error(name() + " IMPLEMENT ME: IndexVar!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Literal> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Literal!");}
    void DefaultIRVisitor::visit(std::shared_ptr<TensorVar> node) { throw std::runtime_error(name() + " IMPLEMENT ME: TensorVar!");}
    void DefaultIRVisitor::visit(std::shared_ptr<TupleVar> node) { throw std::runtime_error(name() + " IMPLEMENT ME: TupleVar!");}
    void DefaultIRVisitor::visit(std::shared_ptr<IndexVarExpr> node) { throw std::runtime_error(name() + " IMPLEMENT ME: IndexVarExpr!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Access> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Access!");}
    void DefaultIRVisitor::visit(std::shared_ptr<ReadAccess> node) { throw std::runtime_error(name() + " IMPLEMENT ME: ReadAccess!");}
    void DefaultIRVisitor::visit(std::shared_ptr<TupleVarReadAccess> node) { throw std::runtime_error(name() + " IMPLEMENT ME: TupleVarReadAccess!");}
    void DefaultIRVisitor::visit(std::shared_ptr<BuiltinFuncDecl> node) { throw std::runtime_error(name() + " IMPLEMENT ME: BuiltinFuncDecl!");}
    void DefaultIRVisitor::visit(std::shared_ptr<BinaryOp> node) { throw std::runtime_error(name() + " IMPLEMENT ME: BinaryOp!");}
    void DefaultIRVisitor::visit(std::shared_ptr<UnaryOp> node) { throw std::runtime_error(name() + " IMPLEMENT ME: UnaryOp!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Definition> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Definition!");}
    void DefaultIRVisitor::visit(std::shared_ptr<MultipleOutputDefinition> node) { throw std::runtime_error(name() + " IMPLEMENT ME: MultipleOutputDefinition!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Allocate> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Allocate!");}
    void DefaultIRVisitor::visit(std::shared_ptr<MemAssignment> node) { throw std::runtime_error(name() + " IMPLEMENT ME: MemAssignment!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Initialize> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Initialize!");}
    void DefaultIRVisitor::visit(std::shared_ptr<FuncDecl> node) { throw std::runtime_error(name() + " IMPLEMENT ME: FuncDecl!");}
    void DefaultIRVisitor::visit(std::shared_ptr<AndOperator> node) { throw std::runtime_error(name() + " IMPLEMENT ME: AndOperator!");}
    void DefaultIRVisitor::visit(std::shared_ptr<OrOperator> node) { throw std::runtime_error(name() + " IMPLEMENT ME: OrOperator!");}
    void DefaultIRVisitor::visit(std::shared_ptr<AddOperator> node) { throw std::runtime_error(name() + " IMPLEMENT ME: AddOperator!");}
    void DefaultIRVisitor::visit(std::shared_ptr<MulOperator> node) { throw std::runtime_error(name() + " IMPLEMENT ME: MulOperator!");}
    void DefaultIRVisitor::visit(std::shared_ptr<MinOperator> node) { throw std::runtime_error(name() + " IMPLEMENT ME: MinOperator!");}
    void DefaultIRVisitor::visit(std::shared_ptr<ChooseOperator> node) { throw std::runtime_error(name() + " IMPLEMENT ME: ChooseOperator!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Call> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Call!");}
    void DefaultIRVisitor::visit(std::shared_ptr<CallStarRepeat> node) { throw std::runtime_error(name() + " IMPLEMENT ME: CallStarRepeat!");}
    void DefaultIRVisitor::visit(std::shared_ptr<CallStarCondition> node) { throw std::runtime_error(name() + " IMPLEMENT ME: CallStarCondition!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Module> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Module!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Reduction> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Reduction!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Datatype> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Datatype!");}
    void DefaultIRVisitor::visit(std::shared_ptr<TensorType> node) { throw std::runtime_error(name() + " IMPLEMENT ME: TensorType!");}
    void DefaultIRVisitor::visit(std::shared_ptr<StorageFormat> node) { throw std::runtime_error(name() + " IMPLEMENT ME: StorageFormat!");}
    void DefaultIRVisitor::visit(std::shared_ptr<TupleType> node) { throw std::runtime_error(name() + " IMPLEMENT ME: TupleType!");}
    void DefaultIRVisitor::visit(std::shared_ptr<Operator> node) { throw std::runtime_error(name() + " IMPLEMENT ME: Operator!");}

    void DefaultIRVisitor::visit(std::shared_ptr<FormatRule> node) {throw std::runtime_error(name() + " IMPLEMENT ME: FormatRule!");}

    std::string TupleVarReadAccess::dump() const {
        return "<" + std::to_string(idx) + ">" + var->dump();
    }

    std::shared_ptr<Type> TupleVarReadAccess::getType() const {
        return var->type;
    }

    std::vector<std::shared_ptr<IndexVar>> TupleVarReadAccess::getIndices() {
        return {};
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>> TupleVarReadAccess::getIndexVarDims(IRContext* context) const {
        return {};
    }

    std::string TupleVar::dump() const {
        return name;
    }

    std::shared_ptr<Type> TupleVar::getType() const {
        return type;
    }

    std::vector<std::shared_ptr<IndexVar>> TupleVar::getIndices() {
        return {};
    }

    std::map<std::string, std::set<std::shared_ptr<Expression>>> TupleVar::getIndexVarDims(IRContext* context) const {
        return {};
    }

    std::string MultipleOutputDefinition::dump() const {
        return lhs->dump() + " = " + rhs->dump();
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<TensorVar> node) {
        node->type->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<TupleVar> node) {
        node->type->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Access> node) {
        node->tensor->accept(this);
        for(auto& idx: node->indices) {
            idx->accept(this);
        }
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<ReadAccess> node) {
        node->tensor->accept(this);
        for (auto& idx: node->indices) {
            idx->accept(this);
        }
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<BinaryOp> node) {
        node->left->accept(this);
        node->right->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<UnaryOp> node) {
        node->expr->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Definition> node) {
        for(auto& acc: node->lhs) {
            acc->accept(this);
        }
        node->rhs->accept(this);
        for(auto& red: node->reduction_list) {
            red->accept(this);
        }
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<MultipleOutputDefinition> node) {
        node->lhs->accept(this);
        node->rhs->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Allocate> node) {
        node->tensor->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<MemAssignment> node) {
        node->lhs->accept(this);
        node->rhs->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Initialize> node) {
        node->tensor->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<FuncDecl> node) {
        for(auto& in: node->inputs) {
            in->accept(this);
        }
        for(auto& out: node->outputs) {
            out->accept(this);
        }
        for(auto& stmt: node->body) {
            stmt->accept(this);
        }
    }

    void DefaultIRVisitorUnsafe::visit_call(std::shared_ptr<Call> node) {
        for(auto& arg: node->arguments) {
            arg->accept(this);
        }
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Call> node) {
        visit_call(node);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<CallStarRepeat> node) {
        visit_call(node);
        for(auto& rule: node->format_rules) {
            rule->accept(this);
        }
    }
    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<CallStarCondition> node) {
        visit_call(node);
        for(auto& rule: node->format_rules) {
            rule->accept(this);
        }
        node->condition_def->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Module> node) {
        for(auto& comp: node->decls) {
            if (comp->is_builtin()) {
                continue;
            }
            if (comp->is_tuple_var()) {
                comp->as_tuple_var()->accept(this);
            }
            if (comp->is_init()) {
                comp->as_init()->accept(this);
            }
            if (comp->is_decl()) {
                comp->as_decl()->accept(this);
            }
            if (comp->is_var()) {
                comp->as_var()->accept(this);
            }
            if (comp->is_def()) {
                comp->as_def()->accept(this);
            }
            if (comp->is_multi_def()) {
                comp->as_multi_def()->accept(this);
            }
            if (comp->is_expr()) {
                comp->as_expr()->accept(this);
            }
            if (comp->is_allocate()) {
                comp->as_allocate()->accept(this);
            }
            if (comp->is_mem_assign()) {
                comp->as_mem_assign()->accept(this);
            }
        }
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Reduction> node) {
        node->reductionVar->accept(this);
        node->reductionInit->accept(this);
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<StorageFormat> node) {}

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<TensorType> node) {
        node->type->accept(this);
        for(auto& dim: node->dimensions) {
            dim->accept(this);
        }
        for(auto& format: node->formats) {
            format->accept(this);
        }
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<TupleType> node) {
        for(auto& t: node->tuple) {
            t->accept(this);
        }
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<IndexVar> node) {
         // std::cout << name() << ": UNIMPLEMENTED IndexVar\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Literal> node) {
         // std::cout << name() << ": UNIMPLEMENTED Literal\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<IndexVarExpr> node) {
         // std::cout << name() << ": UNIMPLEMENTED IndexVarExpr\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<TupleVarReadAccess> node) {
         // std::cout << name() << ": UNIMPLEMENTED TupleVarReadAccess\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<BuiltinFuncDecl> node) {
        node->op->accept(this);
        //TODO: visit inputs and outputs as well + stmt body
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Operator> node) {
         // std::cout << name() << ": UNIMPLEMENTED Operator\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<AndOperator> node) {
         // std::cout << name() << ": UNIMPLEMENTED AndOperator\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<OrOperator> node) {
         // std::cout << name() << ": UNIMPLEMENTED OrOperator\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<AddOperator> node) {
         // std::cout << name() << ": UNIMPLEMENTED AddOperator\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<MulOperator> node) {
         // std::cout << name() << ": UNIMPLEMENTED MulOperator\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<MinOperator> node) {
         // std::cout << name() << ": UNIMPLEMENTED MinOperator\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<ChooseOperator> node) {
         // std::cout << name() << ": UNIMPLEMENTED ChooseOperator";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<Datatype> node) {
         //  // std::cout << name() << ": UNIMPLEMENTED Datatype\n";
    }

    void DefaultIRVisitorUnsafe::visit(std::shared_ptr<FormatRule> node) {
        node->src_tensor->accept(this);
        node->dst_tensor->accept(this);
        node->condition->accept(this);
        node->format_switch_cond->accept(this);
        node->format_switch_def->accept(this);
    }

    bool BuiltinFuncDecl::is_julia_builtin() const {
        return true;
    }

    bool BuiltinFuncDecl::is_finch_builtin() const {
        return false;
    }

    bool OrOperator::is_julia_builtin() const {
        return false;
    }
    bool OrOperator::is_finch_builtin() const {
        return true;
    }
    bool MinOperator::is_julia_builtin() const {
        return true;
    }
    bool MinOperator::is_finch_builtin() const {
        return true;
    }
    bool ChooseOperator::is_julia_builtin() const {
        return false;
    }
    bool ChooseOperator::is_finch_builtin() const {
        return true;
    }

    std::string FormatRule::dump() const {
        return " FormatRule " + src_tensor->name + " " +
                src_tensor->type->dump() + " -> " + dst_tensor->type->dump() +
                " @ " + condition->dump();
    }
}