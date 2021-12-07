/* Mini Calculator */
/* calc.y */

%{
#include "einsum_taco/parser/heading.h"

int yyerror(char *s);
int yylex(void);

using namespace einsum;
%}

%union {
  string*					 op_val;
  string					 *id_val;
  int 					         int_val;
  bool 						 bool_val;
  float 					 float_val;
  einsum::Literal		                 *lit;
  einsum::ArithmeticExpression			 *arith;
  einsum::ComparisonExpression			 *comp;
  einsum::LogicalExpression       		 *logic;
  einsum::ModuloExpression    	   		 *mod;
  einsum::NotExpression           		 *noot;
  einsum::Expression              		 *expression;
  std::vector<std::shared_ptr<einsum::Expression>>    *expr_vec;
  std::vector<std::shared_ptr<einsum::Reduction>>     *reds_vec;
  std::vector<std::shared_ptr<einsum::IndexVar>>     *inds_vec;
  einsum::Definition				*definition;
  einsum::ReadAccess				*r_access;
  einsum::Access				*w_access;
  std::vector<std::shared_ptr<einsum::Access>>     *acc_vec;
  einsum::AddOp					*add_op;
  einsum::MulOp					*mul_op;
  einsum::Reduction 				*red;
}

%start	input

%type	<expression>	orexp
%type	<expression>	andexp
%type	<expression>	eqexp
%type	<expression>	compexp
%type	<expression>	as_exp
%type	<expression>	mdm_exp
%type	<expression>	notexp
%type	<expression> exp
%type   <expr_vec>  access
%type	<expr_vec>  args
%type   <expression> call
%type   <expression> call_star
%type   <inds_vec>   write_access
%type   <red>	     reduction
%type   <reds_vec>   reduction_list
//%type   <op>	     reduction_operator
%type   <r_access>   read_tensor_access
%type   <w_access>   write_tensor_access
%type   <definition> def
%token  <int_val> INTEGER_LITERAL
%token  <bool_val> BOOL_LITERAL
%token  <float_val> FLOAT_LITERAL
%token  <id_val>   IDENTIFIER
%token  <id_val>   TENSOR
%token  <id_val>   COM
%token  <op_val>   PIPE
%token	NOT
%token	MUL
%token	DIV
%token	MOD
%token	PLUS
%token	SUB
%token	GT
%token	GTE
%token	LT
%token	LTE
%token  EQ
%token  NEQ
%token	AND
%token	OR
%token	OPEN_PAREN
%token	CLOSED_PAREN
%token	OPEN_BRACKET
%token	CLOSED_BRACKET
%token  ASSIGN
%token  COLONS
%token EOL


%%

input: /* nothing */
// | input exp EOL  { std::string d = $2->dump(); const char *cstr = d.c_str(); printf("= %s\n", cstr); }
 | input def EOL  { std::string d = $2->dump(); const char *cstr = d.c_str(); printf("= %s\n", cstr); }
 ;

orexp:		andexp
		| orexp OR andexp { $$ = new einsum::LogicalExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::OrOp>());};

andexp:		eqexp | andexp AND eqexp { $$ = new einsum::LogicalExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::AndOp>()); };

eqexp: 		compexp
		| eqexp EQ compexp { $$ = new einsum::ComparisonExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::EqOp>()); }
		| eqexp NEQ compexp { $$ = new einsum::ComparisonExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::NeqOp>()); };

compexp: 	as_exp
		| compexp GT as_exp { $$ = new einsum::ComparisonExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::GtOp>()); }
		| compexp GTE as_exp { $$ = new einsum::ComparisonExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::GteOp>()); }
		| compexp LT as_exp { $$ = new einsum::ComparisonExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::LtOp>()); }
		| compexp LTE as_exp { $$ = new einsum::ComparisonExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::LteOp>()); };

as_exp: 	mdm_exp
		| as_exp PLUS mdm_exp { $$ = new einsum::ArithmeticExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::AddOp>());}
		| as_exp SUB mdm_exp { $$ = new einsum::ArithmeticExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::SubOp>());};

mdm_exp: 	notexp
		| mdm_exp MUL notexp { $$ = new einsum::ArithmeticExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::MulOp>());}
		| mdm_exp DIV notexp { $$ = new einsum::ArithmeticExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::DivOp>());};
		| mdm_exp MOD notexp { $$ = new einsum::ArithmeticExpression(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), std::make_shared<einsum::ModOp>());};

notexp: 	exp
		| NOT notexp { $$ = new einsum::NotExpression(std::shared_ptr<einsum::Expression>($2));};


access:						    {$$ = new std::vector<std::shared_ptr<einsum::Expression>>(); }
| OPEN_BRACKET exp CLOSED_BRACKET  access {
				auto list = new std::vector<std::shared_ptr<einsum::Expression>>();
				list->push_back(std::shared_ptr<einsum::Expression>($2)); list->insert( list->end(), $4->begin(), $4->end());
				$$ = list;
			}
;

read_tensor_access: TENSOR access { $$ = new einsum::ReadAccess(
							std::shared_ptr<einsum::TensorVar>(new einsum::TensorVar(
								*$1,
								std::shared_ptr<einsum::TensorType>(new einsum::TensorType())
							)),
							*$2
						);
				}

write_access:
| OPEN_BRACKET IDENTIFIER CLOSED_BRACKET  write_access {
				auto list = new std::vector<std::shared_ptr<einsum::IndexVar>>();
				list->push_back(std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$2, 0)));
				list->insert( list->end(), $4->begin(), $4->end());
				$$ = list;
			}
write_tensor_access: TENSOR write_access { $$ = new einsum::Access(
								std::shared_ptr<einsum::TensorVar>(new einsum::TensorVar(
									*$1,
									std::shared_ptr<einsum::TensorType>(new einsum::TensorType())
								)),
								*$2
							);
					}

args:			{$$ = new std::vector<std::shared_ptr<einsum::Expression>>(); }
| args COM exp		{$1->push_back(std::shared_ptr<einsum::Expression>($3)); $$ = $1;}

call: IDENTIFIER OPEN_PAREN CLOSED_PAREN	{new einsum::Call(*$1, std::vector<std::shared_ptr<einsum::Expression>>());}
| IDENTIFIER OPEN_PAREN exp args CLOSED_PAREN {$4->push_back(std::shared_ptr<einsum::Expression>($3));
						$$ = new einsum::Call(*$1, *$4);}

call_star: IDENTIFIER MUL OPEN_PAREN CLOSED_PAREN PIPE exp	{new einsum::CallStarCondition(std::shared_ptr<einsum::Expression>($6), *$1, std::vector<std::shared_ptr<einsum::Expression>>());}
| IDENTIFIER MUL OPEN_PAREN exp args CLOSED_PAREN PIPE exp	{
							$5->push_back(std::shared_ptr<einsum::Expression>($4));
							$$ = new einsum::CallStarCondition(std::shared_ptr<einsum::Expression>($8), *$1, *$5);}


exp:		OPEN_PAREN orexp CLOSED_PAREN { $$ = $2;}
		| INTEGER_LITERAL	{ $$ = new einsum::Literal($1, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Int)); }
		| FLOAT_LITERAL		{ $$ = new einsum::Literal($1, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Float)); }
		| BOOL_LITERAL		{ $$ = new einsum::Literal($1, einsum::Type::make<einsum::Datatype>(einsum::Datatype::Kind::Bool)); }
		| IDENTIFIER		{
						$$ = new einsum::IndexVarExpr(std::shared_ptr<einsum::IndexVar>(
								new einsum::IndexVar(*$1, 0)
							)
						)
					}
		| read_tensor_access  { $$ = $1}
		| call	{$$ = $1}
		| call_star	{$$ = $1}
		;
//
//reduction_operator: PLUS 	{$$ = new einsum::AddOp();}
//		    | MUL 	{$$ = new einsum::MulOp();}
//		    | IDENTIFIER {$$ = new einsum::OrOp();}

reduction: 	IDENTIFIER COLONS OPEN_PAREN PLUS COM exp CLOSED_PAREN  {$$ = new einsum::Reduction(
												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
												std::shared_ptr<einsum::AddOp>(new einsum::AddOp()),
												std::shared_ptr<einsum::Expression>($6)
											)}
		| IDENTIFIER COLONS OPEN_PAREN MUL COM exp CLOSED_PAREN  {$$ = new einsum::Reduction(
                												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
                												std::shared_ptr<einsum::MulOp>(new einsum::MulOp()),
                												std::shared_ptr<einsum::Expression>($6)
                											)}
                | IDENTIFIER COLONS OPEN_PAREN IDENTIFIER COM exp CLOSED_PAREN  {$$ = new einsum::Reduction(
                												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
                												std::shared_ptr<einsum::OrOp>(new einsum::OrOp()),
                												std::shared_ptr<einsum::Expression>($6)
                											)}

reduction_list:	{$$ = new std::vector<std::shared_ptr<einsum::Reduction>>();}
| reduction_list COM reduction {$1->push_back(std::shared_ptr<einsum::Reduction>($3)); $$ = $1;}

def:		write_tensor_access ASSIGN exp	{
						auto acc = std::vector<std::shared_ptr<einsum::Access>>();
                                                acc.push_back(std::shared_ptr<einsum::Access>($1));
						$$ = new einsum::Definition(
								acc,
								std::shared_ptr<einsum::Expression>($3),
								std::vector<std::shared_ptr<einsum::Reduction>>()
						)}
		| write_tensor_access ASSIGN exp PIPE reduction reduction_list  {
									auto acc = std::vector<std::shared_ptr<einsum::Access>>();
									acc.push_back(std::shared_ptr<einsum::Access>($1));
									$6->push_back(std::shared_ptr<einsum::Reduction>($5));
									$$ = new einsum::Definition(
											acc,
											std::shared_ptr<einsum::Expression>($3),
											*$6
									)}
%%

int yyerror(string s)
{
  extern int yylineno;	// defined and maintained in lex.c
  extern char *yytext;	// defined and maintained in lex.c

  cerr << "ERROR: " << s << " at symbol \"" << yytext;
  cerr << "\" on line " << yylineno << endl;
  exit(1);
}

int yyerror(char *s)
{
  return yyerror(string(s));
}