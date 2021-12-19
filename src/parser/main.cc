/* main.cc */

#include <einsum_taco/parser/heading.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    FILE* f;
    if ((argc > 1) && ((f = fopen(argv[1], "r")) == NULL))
    {
        cerr << argv[0] << ": File " << argv[1] << " cannot be opened.\n";
        exit( 1 );
    }
    auto module = parse_module(f);
    cout << module.dump();
    return 0;
}
