/* heading.h */

#define YY_NO_UNPUT

#include <iostream>
#include <stdio.h>
#include <string>
#include "einsum_taco/ir/ir.h"
#include "einsum_taco/ir/type.h"

using namespace std;
using namespace einsum;

typedef void * yyscan_t;
struct YYLTYPE;
union YYSTYPE;

struct State {
    yyscan_t scanner;
    einsum::Module* module;
};

int yyerror(State state, const char *);

int yylex(YYSTYPE * yylval_param, yyscan_t scanner);

int yylex_init(yyscan_t*);

void yyset_in(FILE * in_str, yyscan_t scanner);

int yylex_destroy  (yyscan_t yyscanner);

einsum::Module parse_module(FILE* file);

extern "C" int yywrap (yyscan_t yyscanner );

char *yyget_text (yyscan_t yyscanner );

int yyget_lineno (yyscan_t yyscanner );