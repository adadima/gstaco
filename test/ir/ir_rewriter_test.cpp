//
// Created by Alexandra Dima on 10.01.2022.
//

#include <iostream>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir_rewriter.h"
#include "einsum_taco/ir/dump_ast.h"

#include <einsum_taco/parser/heading.h>
#include "../utils.h"

using namespace std;

struct IRTensorChecker : IRMutator {
    IRContext* context;

    explicit IRTensorChecker(IRContext* context) : context(context) {}

    void visit(IndexVar& node) override {}
    void visit(Literal& node) override {}
    void visit(ArithmeticExpression& node) override {}
    void visit(ModuloExpression& node) override {}
    void visit(LogicalExpression& node) override {}
    void visit(ComparisonExpression& node) override {}
    void visit(NotExpression& node) override {}
    void visit(TensorVar& node) override {}
    void visit(IndexVarExpr& node) override {}
    void visit(Access& node) override {}
    void visit(ReadAccess& node) override {}
    void visit(Definition& node) override {}
    void visit(FuncDecl& node) override {}
    void visit(Call& node) override {}
    void visit(CallStarRepeat& node) override {}
    void visit(CallStarCondition& node) override {}
    void visit(Module& node) override {}
    void visit(Reduction& node) override {}

private:
    void visit_binary(const BinaryOp& node) {}
    void visit_unary(const UnaryOp& node) {}
};

class Cleanup : public IRRewriter, public testing::Test {
public:
    DumpAstVisitor* printer;
    Cleanup() : IRRewriter(new IRContext()) {
        printer = new DumpAstVisitor();
    }

    void check_definition(const shared_ptr<Definition>& def, const map<string, shared_ptr<Expression>>& expected) {
        for (auto &[key, value] : def->getIndexVarDims(context)) {

        }
    }

protected:
    virtual void SetUp() {

    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).

    }
};

TEST_F(Cleanup, IndexVars1) {
    const string func = R"(
Let Frontier(edges int[N][N], frontier_list int[N][N], visited int[N], round_in int) -> (frontier int[N], round_out int)
    frontier[j] = edges[j][k] * frontier_list[2][k] * (visited[j] == 0) | k:(OR, 0)
    round_out = round_in * 2
End)";
    auto mod = parse(func);
    mod.accept(this);
    EXPECT_EQ(module->dump(), func + "\n");

    auto ast = R"(<Module
	<Array
			<FuncDecl
				Frontier
				<Array
						<TensorVar edges int[N][N]>
						<TensorVar frontier_list int[N][N]>
						<TensorVar visited int[N]>
						<TensorVar round_in int>
				>
				<Array
						<TensorVar frontier int[N]>
						<TensorVar round_out int>
				>
				<Array
						<Definition
							<Array
									<Access
										<TensorVar frontier int[N]>
										<Array
												<IndexVar
													j
													<ReadAccess
														<TensorVar N int>
														<Array>
													>
												>
										>
									>
							>
							<ArithmeticExpression
								<ArithmeticExpression
									<ReadAccess
										<TensorVar edges int[N][N]>
										<Array
												<IndexVarExpr
													<IndexVar
														j
														<ReadAccess
															<TensorVar N int>
															<Array>
														>
													>
												>
												<IndexVarExpr
													<IndexVar
														k
														<ReadAccess
															<TensorVar N int>
															<Array>
														>
													>
												>
										>
									>
									*
									<ReadAccess
										<TensorVar frontier_list int[N][N]>
										<Array
												<Literal 2 int>
												<IndexVarExpr
													<IndexVar
														k
														<ReadAccess
															<TensorVar N int>
															<Array>
														>
													>
												>
										>
									>
								>
								*
								<ComparisonExpression
									<ReadAccess
										<TensorVar visited int[N]>
										<Array
												<IndexVarExpr
													<IndexVar
														j
														<ReadAccess
															<TensorVar N int>
															<Array>
														>
													>
												>
										>
									>
									==
									<Literal 0 int>
								>
							>
							<Array
									<IndexVar
										k
										<ReadAccess
											<TensorVar N int>
											<Array>
										>
									>
							>
						>
						<Definition
							<Array
									<Access
										<TensorVar round_out int>
										<Array>
									>
							>
							<ArithmeticExpression
								<ReadAccess
									<TensorVar round_in int>
									<Array>
								>
								*
								<Literal 2 int>
							>
							<Array>
						>
				>
			>
	>
>)";
    module->accept(printer);
    //cout << printer->ast;

    EXPECT_EQ(printer->ast, ast);
}
