/* main.cc */

#include <einsum_taco/parser/heading.h>
#include <unistd.h>

// prototype of bison-generated parser function
int yyparse();

//TODO: error checking for pipe and fork
//int parse_from_string(std::string code) {
//    int stdin_pipe[2];
//    int stdout_pipe[2];
//    pipe(stdin_pipe);
//    pipe(stdout_pipe);
//
//    int pid = fork();
//    if (pid == 0) {
//        dup2(stdin_pipe[0], STDIN_FILENO);
//        dup2(stdout_pipe[1],STDOUT_FILENO);
//        const char* sent = code.c_str();
//        write(stdin_pipe[1], sent, strlen(sent));  //TODO: write EOF
//        close(stdin_pipe[1]);
//        yyparse();
//        exit(0);
//    }
//
//    return 0;
//}

//TODO:: use yyrestart
int main(int argc, char **argv)
{
    if ((argc > 1) && (freopen(argv[1], "r", stdin) == NULL))
    {
        cerr << argv[0] << ": File " << argv[1] << " cannot be opened.\n";
        exit( 1 );
    }

    yyparse();

    return 0;
}
