//
// Created by Alexandra Dima on 07.03.2022.
//

#include "taco/tensor.h"

using namespace taco;

int N = 4;
float damp = 0.85;
float beta_score = (1.0 - damp) / N;
Format csr_pr({Dense,Sparse});
Tensor<int> edges_pr("edges", {N, N}, csr_pr);
Tensor<int> out_d("out_d", {N}, Format(Dense));

//Let InitRank() -> (r_out float[N])
//  r_out[j] = 1.0 / N
//End
Tensor<float> InitRank() {
    Tensor<float> R("R", {N}, Format({Dense}));
    IndexVar i;
    R(i) = 1.0 / N;
    R.evaluate();
    std::cout << R << std::endl;
    return R;
}

Tensor<int> OutDegree() {
    Tensor<int> out_d("out_d", {N}, Format(Dense));
    IndexVar i, j;
    out_d(i) = edges_pr(j, i);
    out_d.evaluate();
    std::cout << out_d << std::endl;

    return out_d;
}

//Let PageRankStep(out_d_in int[N], contrib_in float[N], rank_in float[N], r_in float[N]) -> (out_d int[N], contrib float[N], rank float[N], r_out float[N])
//  out_d[j] = edges[i][j] | i:(+, 0)
//  contrib[i] = r_in[i] / out_d[i]
//
//  rank[i] = edges[i][j] * contrib[j] | j:(+, 0.0)
//
//  r_out[i] = beta_score + damp * (rank[i])
//End

std::tuple<Tensor<float>, Tensor<float>, Tensor<float>> PageRankStep(Tensor<float> contrib_in, Tensor<float> rank_in, Tensor<float> r_in) {
    Tensor<float> contrib("contrib", {N}, Format(Dense));
    IndexVar i;
    contrib(i) = r_in(i) / out_d(i);
    contrib.evaluate();
    std::cout << "Contrib: " << contrib << std::endl;

    Tensor<float> rank("rank", {N}, Format(Dense));
    IndexVar j;
    rank(i) = edges_pr(i, j) * contrib(j);
    rank.evaluate();
    std::cout << "Rank: " << rank << std::endl;

    Tensor<float> r_out("r_out", {N}, Format(Dense));
    r_out(i) = beta_score + damp * rank(i);
    r_out.evaluate();
    std::cout << "r_out: " << r_out << std::endl;

    return {contrib, rank, r_out};
}

//
//Let PageRank() -> (out_d int[N], contrib float[N], rank float[N], r_out float[N])
//  out_d[i] = 0
//  contrib[i] = 0.0
//  rank[i] = 0.0
//  _, _, _, r_out = PageRankStep*(out_d, contrib, rank, InitRank()) | 20
//End
Tensor<float> PageRank() {
    Tensor<float> contrib_in("contrib_in", {N}, Format(Dense));
    Tensor<float> rank("rank", {N}, Format(Dense));

    IndexVar i;
    contrib_in(i) = Literal::zero(Float64);
    contrib_in.evaluate();
//    std::cout << "Contrib in: " << contrib_in << std::endl;

    rank(i) = Literal::zero(Float64);
    rank.evaluate();
//    std::cout << "Rank: " << rank << std::endl;

    auto r_in = InitRank();

    for(int j=0; j < 20; j++) {
        auto out = PageRankStep(contrib_in, rank, r_in);
        contrib_in = std::get<0>(out);
        rank = std::get<1>(out);
        r_in = std::get<2>(out);
    }

    return r_in;
}

int main(int argc, char* argv[]) {

    edges_pr.insert({1, 0}, 1);
    edges_pr.insert({2, 0}, 1);
    edges_pr.insert({2, 1}, 1);
    edges_pr.insert({0, 2}, 1);
    edges_pr.insert({2, 3}, 1);
    edges_pr.pack();

    out_d = OutDegree();

    auto ranks = PageRank();
    std::cout << "Ranks: " << ranks << std::endl;
}