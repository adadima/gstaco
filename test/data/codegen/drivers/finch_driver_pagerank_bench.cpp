//
// Created by Alexandra Dima on 30.10.2022.
//

#include "../../../tmp/codegen/pagerank.h"
#include <string>
#include <fstream>
#include "julia.h"
#include <iostream>
#include <unistd.h>

int N;
float damp;
float beta_score;
jl_value_t* edges;

int main(int argc, char* argv[]) {
    auto pid = getpid();
    auto starter = argv[2];
    auto graph = argv[1];
    damp = atof(argv[3]); // 0.85
    enter_finch();
    compile();
    printf("Compiled finch\n");

    Graph g = Graph{};
    N = make_weights_and_edges(starter, &g);
    edges = g.edges;
    beta_score = (1.0 - damp) / N;
    Main();
    printf("Ran starter\n");

    g = Graph{};
    N = make_weights_and_edges(graph, &g);
    printf("Loaded edges\n");
    edges = g.edges;
    beta_score = (1.0 - damp) / N;

    std::ofstream myfile;
    myfile.open ("pagerank_benchmarks.csv", std::ios::out | std::ios::app);
//    myfile << "N,time\n";
    float avg;
    for (int i=0; i < 20; i++) {
        time_t start = time(0);
        Main();
        time_t end = time(0);
        myfile << pid << ", " << N << "," << (end-start) << "\n";
        avg += end - start;
        printf("[PAGERANK] Finished run %d on %s in %ld seconds\n", i, argv[1], (end - start));
    }
    avg /= 20;
    printf("[PAGERANK] Average time for N=%d : %lf\n", N, avg);

    myfile.close();

    exit_finch();
}