%{
#include "einsum_taco/parser/heading.h"
%}


%parse-param {State state}
%lex-param {State state}

%union {
  string*					 op_val;
  string					 *id_val;
  int 					         int_val;
  bool 						 bool_val;
  float 					 float_val;
  einsum::Literal		                 *lit;
  einsum::BinaryOp				*binary;
  einsum::UnaryOp				*unary;
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
  einsum::StorageFormat				*sformat;
  einsum::FuncDecl				*fdecl;
  einsum::BuiltinFuncDecl			*builtindecl;
  std::vector<std::shared_ptr<einsum::Statement>>	*stmt_vec;
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
%type   <builtindecl>     builtin
%type   <expression> call
%type   <expression> call_star
%type   <expression> call_repeat
%type   <inds_vec>   write_access
%type   <red>	     reduction
%type   <reds_vec>   reduction_list
%type   <ttype> 	type
%type   <sformat> 	format
%type   <fdecl> 	func
%type   <stmt_vec>      statements
%type   <tvar_vec> 	input_params
%type   <tvar_vec>	output_params
%type   <tvar_vec>	param_list
%type   <tvar> 		param
%type   <acc_vec>       def_lhs
%type   <r_access>   read_tensor_access
%type   <w_access>   write_tensor_access
%type   <definition> def
%type   <tvar> 	tensor
%token  <int_val> INTEGER_LITERAL
%token  <bool_val> BOOL_LITERAL
%token  <float_val> FLOAT_LITERAL
%token  <id_val>   LET
%token  <id_val>   END
%token  <id_val>   RARROW
%token  <id_val>   IDENTIFIER
%token  <id_val>   STAR_CALL
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
%token	MIN
%token	CHOOSE
%token  OR_RED
%token  AND_RED
%token	OPEN_PAREN
%token	CLOSED_PAREN
%token	OPEN_BRACKET
%token	CLOSED_BRACKET
%token  DENSE
%token  SPARSE
%token  ASSIGN
%token  COLONS
%token EOL


%%
// func EOL blank EOL def EOL blank EOL orexp EOL
input: | blank
       | input func blank  {state.module->add(std::shared_ptr<ModuleComponent>($2));}
       | input orexp blank {state.module->add(std::shared_ptr<ModuleComponent>($2));}
       | input def blank   {state.module->add(std::shared_ptr<ModuleComponent>($2));}
       | input tensor blank {state.module->add(std::shared_ptr<ModuleComponent>($2));}
 ;

blank:
| blank EOL

orexp:		andexp
		| orexp OR andexp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::or_, einsum::boolType);};

andexp:		eqexp | andexp AND eqexp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::and_, einsum::boolType); };

eqexp: 		compexp
		| eqexp EQ compexp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3),einsum::eq, einsum::boolType); }
		| eqexp NEQ compexp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::neq, einsum::boolType); };

compexp: 	as_exp
		| compexp GT as_exp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::gt, einsum::boolType); }
		| compexp GTE as_exp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::gte, einsum::boolType); }
		| compexp LT as_exp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::lt, einsum::boolType); }
		| compexp LTE as_exp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::lte, einsum::boolType); };

as_exp: 	mdm_exp
		| as_exp PLUS mdm_exp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::add, nullptr);}
		| as_exp SUB mdm_exp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::sub, nullptr);};

mdm_exp: 	notexp
		| mdm_exp MUL notexp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::mul, nullptr);}
		| mdm_exp DIV notexp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::div, nullptr);};
		| mdm_exp MOD notexp { $$ = new einsum::BinaryOp(std::shared_ptr<einsum::Expression>($1), std::shared_ptr<einsum::Expression>($3), einsum::mod, einsum::intType);};

notexp: 	exp
		| NOT notexp { $$ = new einsum::UnaryOp(std::shared_ptr<einsum::Expression>($2), einsum::not_, einsum::boolType);};


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
									std::shared_ptr<einsum::TensorType>(new einsum::TensorType()),
									false
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
							std::shared_ptr<einsum::TensorType>(new einsum::TensorType()),
							false
						)),
						*$2
					);
					$4->insert($4->begin(), std::shared_ptr<einsum::Access>(acc));
					$$ = $4;}

args: orexp			{auto args = new std::vector<std::shared_ptr<einsum::Expression>>(); args->push_back(std::shared_ptr<einsum::Expression>($1)); $$ = args;}
| orexp COM args		{auto args = new std::vector<std::shared_ptr<einsum::Expression>>();
				args->push_back(std::shared_ptr<einsum::Expression>($1));
				args->insert(args->end(), $3->begin(), $3->end());
				$$ = args;}

builtin: CHOOSE {$$ = einsum::choose_red.get();}
| MIN {$$ = einsum::min_red.get();}

call: IDENTIFIER OPEN_PAREN CLOSED_PAREN	{$$ = new einsum::Call(*$1, std::vector<std::shared_ptr<einsum::Expression>>());}
| IDENTIFIER OPEN_PAREN args CLOSED_PAREN   {$$ = new einsum::Call(*$1, *$3);}
| builtin OPEN_PAREN CLOSED_PAREN           {$$ = new einsum::Call(std::shared_ptr<einsum::BuiltinFuncDecl>($1), std::vector<std::shared_ptr<einsum::Expression>>());}
| builtin OPEN_PAREN args CLOSED_PAREN      {$$ = new einsum::Call(std::shared_ptr<einsum::BuiltinFuncDecl>($1), *$3);}

call_repeat: STAR_CALL OPEN_PAREN CLOSED_PAREN PIPE INTEGER_LITERAL	{$1->pop_back(); new einsum::CallStarRepeat($5, *$1, std::vector<std::shared_ptr<einsum::Expression>>());}
             | STAR_CALL OPEN_PAREN args CLOSED_PAREN PIPE INTEGER_LITERAL	{ $1->pop_back();
             							$$ = new einsum::CallStarRepeat($6, *$1, *$3);}


call_star: STAR_CALL OPEN_PAREN CLOSED_PAREN PIPE orexp	{$1->pop_back(); new einsum::CallStarCondition(std::shared_ptr<einsum::Expression>($5), *$1, std::vector<std::shared_ptr<einsum::Expression>>());}
| STAR_CALL OPEN_PAREN args CLOSED_PAREN PIPE orexp	{ $1->pop_back();
							$$ = new einsum::CallStarCondition(std::shared_ptr<einsum::Expression>($6), *$1, *$3);}


exp:		OPEN_PAREN orexp CLOSED_PAREN { $$ = $2;}
		| INTEGER_LITERAL	{ $$ = new einsum::Literal($1, einsum::intType); }
		| FLOAT_LITERAL		{ $$ = new einsum::Literal($1, einsum::floatType); }
		| BOOL_LITERAL		{ $$ = new einsum::Literal($1, einsum::boolType); }
		| IDENTIFIER access 	{ $$ = new einsum::ReadAccess(
								std::shared_ptr<einsum::TensorVar>(new einsum::TensorVar(
									*$1,
									std::shared_ptr<einsum::TensorType>(new einsum::TensorType()),
									false
								)),
								*$2
							);
					}
		| call	{$$ = $1;}
		| call_repeat	{$$ = $1;}
		| call_star	{$$ = $1;}
		;

reduction: 	IDENTIFIER COLONS OPEN_PAREN PLUS COM orexp CLOSED_PAREN  {$$ = new einsum::Reduction(
												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
												einsum::add_red,
												std::shared_ptr<einsum::Expression>($6)
											);}
		| IDENTIFIER COLONS OPEN_PAREN MUL COM orexp CLOSED_PAREN  {$$ = new einsum::Reduction(
                												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
                												einsum::mul_red,
                												std::shared_ptr<einsum::Expression>($6)
                											);}
                | IDENTIFIER COLONS OPEN_PAREN AND_RED COM orexp CLOSED_PAREN  {$$ = new einsum::Reduction(
                                												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
                                												einsum::and_red,
                                												std::shared_ptr<einsum::Expression>($6)
                                											);}
                | IDENTIFIER COLONS OPEN_PAREN OR_RED COM orexp CLOSED_PAREN  {$$ = new einsum::Reduction(
                                												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
                                												einsum::or_red,
                                												std::shared_ptr<einsum::Expression>($6)
                                											);}
                | IDENTIFIER COLONS OPEN_PAREN MIN COM orexp CLOSED_PAREN  {$$ = new einsum::Reduction(
                												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
                												einsum::min_red,
                												std::shared_ptr<einsum::Expression>($6)
                											);}
                | IDENTIFIER COLONS OPEN_PAREN CHOOSE COM orexp CLOSED_PAREN  {$$ = new einsum::Reduction(
                                												std::shared_ptr<einsum::IndexVar>(new einsum::IndexVar(*$1, 0)),
                                												einsum::choose_red,
                                												std::shared_ptr<einsum::Expression>($6)
                                											);}

reduction_list:	{$$ = new std::vector<std::shared_ptr<einsum::Reduction>>();}
| reduction_list COM reduction {$1->push_back(std::shared_ptr<einsum::Reduction>($3)); $$ = $1;}

def:		def_lhs ASSIGN orexp	{	$$ = new einsum::Definition(
								*$1,
								std::shared_ptr<einsum::Expression>($3),
								std::vector<std::shared_ptr<einsum::Reduction>>()
						);}
		| def_lhs ASSIGN orexp PIPE reduction reduction_list  {
									$6->insert($6->begin(), std::shared_ptr<einsum::Reduction>($5));
									$$ = new einsum::Definition(
											*$1,
											std::shared_ptr<einsum::Expression>($3),
											*$6
									);}
format: DENSE  {
               			$$ = new einsum::StorageFormat(Dense);
               	}
| SPARSE		{
				$$ = new einsum::StorageFormat(Sparse);
			}

type: IDENTIFIER				{
						auto dims = std::vector<std::shared_ptr<einsum::Expression>>();
						auto formats = std::vector<std::shared_ptr<einsum::StorageFormat>>();
						$$ = new einsum::TensorType(std::make_shared<einsum::Datatype>(*$1), dims, formats);}
| type OPEN_BRACKET orexp CLOSED_BRACKET		{
						 auto dims = std::vector<std::shared_ptr<einsum::Expression>>();
						 for (int i=0; i < $1->getDimensions().size(); i++) {
							dims.push_back(($1->getDimensions())[i]);
						 }
						 dims.push_back(std::shared_ptr<einsum::Expression>($3));
						 $$ = new einsum::TensorType($1->getElementType(), dims);
						}
| type OPEN_BRACKET format OPEN_BRACKET orexp CLOSED_BRACKET CLOSED_BRACKET {
						 auto dims = std::vector<std::shared_ptr<einsum::Expression>>();
						 for (int i=0; i < $1->getDimensions().size(); i++) {
							dims.push_back(($1->getDimensions())[i]);
						 }
						 dims.push_back(std::shared_ptr<einsum::Expression>($5));

						 auto forms = std::vector<std::shared_ptr<einsum::StorageFormat>>();
						 for (int i=0; i < $1->getDimensions().size(); i++) {
						 	forms.push_back($1->formats[i]);
						 }
						 forms.push_back(std::shared_ptr<einsum::StorageFormat>($3));
						 $$ = new einsum::TensorType($1->getElementType(), dims, forms);
}

param: IDENTIFIER type				{assert($2 != nullptr);
$$ = new einsum::TensorVar(*$1, std::shared_ptr<einsum::TensorType>($2), false);}

param_list:					{$$ = new std::vector<std::shared_ptr<einsum::TensorVar>>();}
| COM param param_list				{$3->insert($3->begin(), std::shared_ptr<einsum::TensorVar>($2));
                                                 $$ = $3;}

input_params: OPEN_PAREN CLOSED_PAREN		{$$ = new std::vector<std::shared_ptr<einsum::TensorVar>>();}
| OPEN_PAREN param param_list CLOSED_PAREN	{$3->insert($3->begin(), std::shared_ptr<einsum::TensorVar>($2));
						 $$ = $3;}

output_params:					{$$ = new std::vector<std::shared_ptr<einsum::TensorVar>>();}
| OPEN_PAREN param param_list CLOSED_PAREN	{$3->insert($3->begin(), std::shared_ptr<einsum::TensorVar>($2));
                                                 $$ = $3;}

statements:					{$$ = new std::vector<std::shared_ptr<einsum::Statement>>();}
|  def EOL blank statements				{$4->insert($4->begin(), std::shared_ptr<einsum::Statement>($1));
						 $$ = $4;}

func:		LET IDENTIFIER input_params RARROW output_params EOL blank statements END {$$ = new einsum::FuncDecl(*$2, *$3, *$5, *$8);}

tensor:	IDENTIFIER type				{
assert($2 != nullptr);
$$ = new einsum::TensorVar(*$1, std::shared_ptr<einsum::TensorType>($2), true);}
%%

int yyerror(State state, string s)
{
  std::stringstream sin;
  sin << "ERROR: " << s << " at symbol \"" << yyget_text(state.scanner);
  sin << "\" on line " << yyget_lineno(state.scanner);
  const auto msg = sin.str();
  throw std::runtime_error(msg);
}

int yyerror(State state, const char *s)
{
  return yyerror(state, string(s));
}

extern "C" int yywrap(yyscan_t scanner) {
  /* Since this is a simple demonstration, so we will just terminate when we reach the end of the input file */
  return 1;
}

einsum::Module parse_module(FILE* file) {
	auto module = einsum::Module(std::vector<std::shared_ptr<einsum::ModuleComponent>>());
	yyscan_t scanner;
	yylex_init(&scanner) ;
	auto state = State{scanner, &module};

	yyset_in(file, scanner);

	int status;
	yypstate *ps = yypstate_new ();
	YYSTYPE pushed_value;

	do {
	    auto l = yylex(&pushed_value, scanner);
	    status = yypush_parse(ps, l, &pushed_value, state);
	} while(status == YYPUSH_MORE);

	yypstate_delete (ps);
	yylex_destroy (scanner) ;
	return module;
}