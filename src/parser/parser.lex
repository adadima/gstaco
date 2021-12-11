/* Mini Calculator */
/* calc.lex */

%{
#include <einsum_taco/parser/heading.h>
#include "tok.h"
int yyerror(char *s);
%}

%option noyywrap

digit		[0-9]

int_const	{digit}+

ID       [a-zA-Z_#][a-zA-Z_#0-9]*

%%
{int_const}	        {yylval.int_val = atoi(yytext); return INTEGER_LITERAL; }

{digit}+"."+{digit}*  {yylval.float_val = std::stof(yytext); return FLOAT_LITERAL; }

"!"                 { yylval.op_val = new std::string(yytext); return NOT; }
"*"                 { yylval.op_val = new std::string(yytext); return MUL; }
"/"                 { yylval.op_val = new std::string(yytext); return DIV; }
"%"                 { yylval.op_val = new std::string(yytext); return MOD; }
"+"                 { yylval.op_val = new std::string(yytext); return PLUS; }
"-"                 { yylval.op_val = new std::string(yytext); return SUB; }
">"                 { yylval.op_val = new std::string(yytext); return GT; }
">="                { yylval.op_val = new std::string(yytext); return GTE; }
"<"                 { yylval.op_val = new std::string(yytext); return LT; }
"<="                { yylval.op_val = new std::string(yytext); return LTE; }
"=="                { yylval.op_val = new std::string(yytext); return EQ; }
"!="                { yylval.op_val = new std::string(yytext); return NEQ; }
"&&"                { yylval.op_val = new std::string(yytext); return AND; }
"||"		        { yylval.op_val = new std::string(yytext); return OR; }

"="      { yylval.op_val = new std::string(yytext); return ASSIGN; }
"("      { yylval.op_val = new std::string(yytext); return OPEN_PAREN; }
")"      { yylval.op_val = new std::string(yytext); return CLOSED_PAREN; }
"["      { yylval.op_val = new std::string(yytext); return OPEN_BRACKET; }
"]"      { yylval.op_val = new std::string(yytext); return CLOSED_BRACKET; }
":"      { yylval.op_val = new std::string(yytext); return COLONS; }
","      { yylval.id_val = new std::string(yytext); return COM; }

"|"      { yylval.op_val = new std::string(yytext); return PIPE; }

true|false    {yylval.bool_val = (std::string(yytext) == "true") ? true : false; return BOOL_LITERAL;}

"Let"       {yylval.id_val = new std::string(yytext); return LET; }

"End"       {yylval.id_val = new std::string(yytext); return END; }


if|then|else {
            printf( "A keyword: %s\n", yytext );
            }

_|{ID}        {yylval.id_val = new std::string(yytext); return IDENTIFIER; }

"->"        {yylval.id_val = new std::string(yytext); return RARROW; }

[ \t]*          /* eat up whitespace */

[\n]		{ yylineno++;	return EOL;}

"{"[^}\n]*"}"     /* eat up one-line comments */

.		{ std::cerr << "SCANNER "; yyerror(""); exit(1);	}
