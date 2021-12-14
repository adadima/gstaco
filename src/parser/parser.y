/* Mini Calculator */
/* calc.y */

%{
#include "einsum_taco/parser/heading.h"

int yyerror(char *s);
int yylex(void);

using namespace einsum;
%}

%parse-param {int *nastiness} {int *randomness}

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
  einsum::TensorType				*ttype;
  einsum::FuncDecl				*fdecl;
  std::vector<std::shared_ptr<einsum::Definition>>	*defs_vec;
  std::vector<std::shared_ptr<einsum::TensorVar>>	*tvar_vec;
  einsum::TensorVar				*tvar;
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
%type   <ttype> 	type
%type   <fdecl> 	func
%type   <defs_vec>      statements
%type   <tvar_vec> 	input_params
%type   <tvar_vec>	output_params
%type   <tvar_vec>	param_list
%type   <tvar> 		param
%type   <acc_vec>       def_lhs
//%type   <op>	     reduction_operator
%type   <r_access>   read_tensor_access
%type   <w_access>   write_tensor_access
%type   <definition> def
%token  <int_val> INTEGER_LITERAL
%token  <bool_val> BOOL_LITERAL
%token  <float_val> FLOAT_LITERAL
%token  <id_val>   LET
%token  <id_val>   END
%token  <id_val>   RARROW
%token  <id_val>   IDENTIFIER
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

input: | blank
       | input func EOL blank  { std::string d = $2->dump(); const char *cstr = d.c_str(); printf("%s\n", cstr); }
// | input orexp EOL  { std::string d = $2->dump(); const char *cstr = d.c_str(); printf("= %s\n", cstr); }
// | input def EOL  { std::string d = $2->dump(); const char *cstr = d.c_str(); printf("= %s\n", cstr); }
 ;

blank:
| blank EOL

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
| OPEN_BRACKET orexp CLOSED_BRACKET  access {
				auto list = new std::vector<std::shared_ptr<einsum::Expression>>();
				list->push_back(std::shared_ptr<einsum::Expression>($2));
				list->insert( list->end(), $4->begin(), $4->end());
				$$ = list;
			}
;

read_tensor_access: IDENTIFIER access { $$ = new einsum::ReadAccess(
							std::shared_ptr<einsum::TensorVar>(new einsum::TensorVar(
								*$1,
								std::shared_ptr<einsum::TensorType>(new einsum::TensorType())
							)),
							*$2
						);
				}

write_access:					{$$ = new std::vector<std::shared_ptr<einsum::IndexVar>>(); }
| OPEN_BRACKET IDENTIFIER CLOSED_BRACKET  write_access {
				auto list = new std::vector<std::shared_ptr<einsum::IndexVar>>();
				list->push_back(std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$2, 0)));
				list->insert( list->end(), $4->begin(), $4->end());
				$$ = list;
			}
write_tensor_access: IDENTIFIER write_access { $$ = new einsum::Access(
								std::shared_ptr<einsum::TensorVar>(new einsum::TensorVar(
									*$1,
									std::shared_ptr<einsum::TensorType>(new einsum::TensorType())
								)),
								*$2
							);
					}

def_lhs: IDENTIFIER write_access		{auto acc =  new einsum::Access(
								std::shared_ptr<einsum::TensorVar>(new einsum::TensorVar(
									*$1,
									std::shared_ptr<einsum::TensorType>(new einsum::TensorType())
								)),
								*$2
							);
					auto outputs = new std::vector<std::shared_ptr<einsum::Access>>();
					outputs->push_back(std::shared_ptr<einsum::Access>(acc));
					$$ = outputs;}
| IDENTIFIER write_access COM def_lhs	{
					auto acc =  new einsum::Access(
						std::shared_ptr<einsum::TensorVar>(new einsum::TensorVar(
							*$1,
							std::shared_ptr<einsum::TensorType>(new einsum::TensorType())
						)),
						*$2
					);
					$4->insert($4->begin(), std::shared_ptr<einsum::Access>(acc));
					$$ = $4;}

args:			{$$ = new std::vector<std::shared_ptr<einsum::Expression>>(); }
| args COM orexp		{$1->push_back(std::shared_ptr<einsum::Expression>($3)); $$ = $1;}

call: IDENTIFIER OPEN_PAREN CLOSED_PAREN	{new einsum::Call(*$1, std::vector<std::shared_ptr<einsum::Expression>>());}
| IDENTIFIER OPEN_PAREN orexp args CLOSED_PAREN   {$4->insert($4->begin(), std::shared_ptr<einsum::Expression>($3));
						$$ = new einsum::Call(*$1, *$4);}

call_star: IDENTIFIER MUL OPEN_PAREN CLOSED_PAREN PIPE orexp	{new einsum::CallStarCondition(std::shared_ptr<einsum::Expression>($6), *$1, std::vector<std::shared_ptr<einsum::Expression>>());}
| IDENTIFIER MUL OPEN_PAREN orexp args CLOSED_PAREN PIPE orexp	{
							$5->insert($5->begin(), std::shared_ptr<einsum::Expression>($4));
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
		| IDENTIFIER access 	{ $$ = new einsum::ReadAccess(
								std::shared_ptr<einsum::TensorVar>(new einsum::TensorVar(
									*$1,
									std::shared_ptr<einsum::TensorType>(new einsum::TensorType())
								)),
								*$2
							);
					}
		| call	{$$ = $1}
		| call_star	{$$ = $1}
		;
//
//reduction_operator: PLUS 	{$$ = new einsum::AddOp();}
//		    | MUL 	{$$ = new einsum::MulOp();}
//		    | IDENTIFIER {$$ = new einsum::OrOp();}

reduction: 	IDENTIFIER COLONS OPEN_PAREN PLUS COM orexp CLOSED_PAREN  {$$ = new einsum::Reduction(
												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
												std::shared_ptr<einsum::AddOp>(new einsum::AddOp()),
												std::shared_ptr<einsum::Expression>($6)
											)}
		| IDENTIFIER COLONS OPEN_PAREN MUL COM orexp CLOSED_PAREN  {$$ = new einsum::Reduction(
                												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
                												std::shared_ptr<einsum::MulOp>(new einsum::MulOp()),
                												std::shared_ptr<einsum::Expression>($6)
                											)}
                | IDENTIFIER COLONS OPEN_PAREN IDENTIFIER COM orexp CLOSED_PAREN  {$$ = new einsum::Reduction(
                												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
                												std::shared_ptr<einsum::OrOp>(new einsum::OrOp()),
                												std::shared_ptr<einsum::Expression>($6)
                											)}

reduction_list:	{$$ = new std::vector<std::shared_ptr<einsum::Reduction>>();}
| reduction_list COM reduction {$1->push_back(std::shared_ptr<einsum::Reduction>($3)); $$ = $1;}

def:		def_lhs ASSIGN orexp	{
						$$ = new einsum::Definition(
								*$1,
								std::shared_ptr<einsum::Expression>($3),
								std::vector<std::shared_ptr<einsum::Reduction>>()
						)}
		| def_lhs ASSIGN orexp PIPE reduction reduction_list  {
									$6->insert($6->begin(), std::shared_ptr<einsum::Reduction>($5));
									$$ = new einsum::Definition(
											*$1,
											std::shared_ptr<einsum::Expression>($3),
											*$6
									)}
type: IDENTIFIER				{
						auto dims = std::vector<std::shared_ptr<einsum::Expression>>();
						$$ = new einsum::TensorType(std::make_shared<einsum::Datatype>(*$1), dims);}
| type OPEN_BRACKET orexp CLOSED_BRACKET		{
						 auto dims = std::vector<std::shared_ptr<einsum::Expression>>();
						 for (int i=0; i < $1->getDimensions().size(); i++) {
							dims.push_back(($1->getDimensions())[i]);
						 }
						 dims.push_back(std::shared_ptr<einsum::Expression>($3));
						 $$ = new einsum::TensorType($1->getElementType(), dims);;
						}

param: IDENTIFIER type				{$$ = new einsum::TensorVar(*$1, std::shared_ptr<einsum::TensorType>($2));}

param_list:					{$$ = new std::vector<std::shared_ptr<einsum::TensorVar>>();}
| COM param param_list				{$3->insert($3->begin(), std::shared_ptr<einsum::TensorVar>($2));
                                                 $$ = $3;}

input_params: OPEN_PAREN CLOSED_PAREN		{$$ = new std::vector<std::shared_ptr<einsum::TensorVar>>();}
| OPEN_PAREN param param_list CLOSED_PAREN	{$3->insert($3->begin(), std::shared_ptr<einsum::TensorVar>($2));
						 $$ = $3;}

output_params:					{$$ = new std::vector<std::shared_ptr<einsum::TensorVar>>();}
| OPEN_PAREN param param_list CLOSED_PAREN	{$3->insert($3->begin(), std::shared_ptr<einsum::TensorVar>($2));
                                                 $$ = $3;}

statements:					{$$ = new std::vector<std::shared_ptr<einsum::Definition>>();}
|  def EOL blank statements				{$4->insert($4->begin(), std::shared_ptr<einsum::Definition>($1));
						 $$ = $4;}

func:		LET IDENTIFIER input_params RARROW output_params EOL blank statements END {$$ = new einsum::FuncDecl(*$2, *$3, *$5, *$8);}
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