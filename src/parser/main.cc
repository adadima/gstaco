/* main.cc */

#include <einsum_taco/parser/heading.h>
#include <unistd.h>

// prototype of bison-generated parser function
int yyparse(vector<einsum::FuncDecl> *declarations);


int main(int argc, char **argv)
{
    if ((argc > 1) && (freopen(argv[1], "r", stdin) == NULL))
    {
        cerr << argv[0] << ": File " << argv[1] << " cannot be opened.\n";
        exit( 1 );
    }
    auto declarations = vector<einsum::FuncDecl>();
    yyparse(&declarations);
    for (const auto & declaration : declarations) {
        cout << declaration.dump();
    }

    return 0;
}
