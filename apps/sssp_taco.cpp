//
// Created by Alexandra Dima on 17.03.2022.
//
#include "taco/tensor.h"
#include "custom_ops.h"
#include "taco/index_notation/kernel.h"

int N = 4;

int P = 1000;

auto source = Literal(0);
//
Format csr({Dense,Sparse});
Tensor<int> edges("edges", {N, N}, csr);
Tensor<double> weights("weights", {N, N}, csr);
Tensor<int> v("v", {N}, Format({Dense}));
Tensor<int> vP("v", {P}, Format({Dense}));

struct pQInit {
    Datatype type_;
    pQInit(Datatype type_) : type_(type_) {}

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        taco_iassert(v.size() == 8);
        return ir::Add::make(
                ir::Cast::make(
                    ir::And::make(
                            ir::Eq::make(v[0], v[1]),
                            ir::Eq::make(v[2], v[3])
                            ),
                            type_
                            ),
                ir::Cast::make(
                        ir::And::make(
                                ir::Eq::make(v[0], v[4]),
                                ir::Neq::make(v[2], v[3])
                        ),
                        type_
                )
                );
    }
};

Func pq_init("pq_init", pQInit(Int64), FullSpace());

//Let Init(source int) -> (dist float[N], priorityQ int[P][N])
//        dist[j] = (j != source) * P
//        priorityQ[p][j] = (p == 0 && j == source) + (p == (P - 1) && j != source)
//End
std::tuple<Tensor<double>, Tensor<int>> Init() {
    Tensor<double> dist("dist", {N}, Format({Dense}));
    IndexVar i, j;

    dist(i) = Cast(neq(v(i), source), Float64) * P;
    dist.evaluate();
    std::cout << dist << std::endl;

    Tensor<int> pQ("priorityQ", {P, N}, Format({Dense, Dense}));
    auto stmt = forall(i,
                       forall(j,
                              Assignment(pQ(i, j), pq_init(i, 0, j, source, P-1))
                              ));
//    pQ(i, j) = pq_init(i, 0, j, source, P-1);
//    pQ.evaluate();

    Kernel k = compile(stmt);
    k(pQ.getStorage());
    pQ.setStorage(pQ.getStorage());
    std::cout << pQ << std::endl;

    return {dist, pQ};
}

struct newDistOp {
    Datatype type_;
    newDistOp(Datatype type_) : type_(type_) {}

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        taco_iassert(v.size() == 8);
        return ir::Add::make(
                    ir::Mul::make(
                            ir::Cast::make(
                                    ir::Eq::make(
                                            ir::Mul::make(v[4], v[5]),
                                            v[6]
                                            ),
                                            type_
                                    ),
                                    v[7]
                    ),
                    ir::Mul::make(
                            ir::Mul::make(v[0], v[1]),
                            ir::Add::make(v[2], v[3])
                            )

        );
    }
};

Func new_dist_op("new_dist_op", newDistOp(Float64), FullSpace());

//  new_dist[j] = edges[j][k] * priorityQ[priority][k] * (weights[j][k] + dist[k]) + (edges[j][k] * priorityQ[priority][k] == 0) * P | k:(MIN, dist[j])
Tensor<double> New_Dist(Tensor<double> dist, Tensor<int> pQ, int priority) {
    Tensor<double> new_dist("new_dist", {N}, Format({Dense}));
    Tensor<double> tmp("tmp", {N}, Format({Dense}));
    IndexVar j, k, p;

    auto stmt = Sequence(
            forall(j,
                   new_dist(j) = dist(j)
            ),
            where(
                    forall(j,
                           forall(k,
                                  Assignment(
                                          new_dist(j),
                                          new_dist_op(
                                                  edges(j, k),
                                                  tmp(k),
                                                  weights(j, k),
                                                  dist(k),
                                                  edges(j, k),
                                                  tmp(k),
                                                  0,
                                                  P
                                          ),
                                          customMin()
                                  )
                           )
                    ),
                    forall(k,
                           forall(p,
                                  tmp(k) = Cast(eq(vP(p), priority), Int64) * pQ(p, k)
                                  ))
                    )

    );
    auto kernel = compile(stmt);
    // std::cout << kernel << std::endl;
    kernel(new_dist.getStorage(), dist.getStorage(), v.getStorage(), pQ.getStorage(), edges.getStorage(), weights.getStorage());
    new_dist.setStorage(new_dist.getStorage());
    // std::cout << new_dist << std::endl;
    return new_dist;
}

//Let UpdateEdges(dist float[N], priorityQ int[P][N], priority int) -> (new_dist float[N], new_priorityQ int[P][N], new_priority int)
//  new_dist[j] = edges[j][k] * priorityQ[priority][k] * (weights[j][k] + dist[k]) + (edges[j][k] * priorityQ[priority][k] == 0) * P | k:(MIN, dist[j])
//  new_priorityQ[j][k] = (dist[k] > new_dist[k]) * (j <= new_dist[k] &&  new_dist[k] < j + 1) + (dist[k] == new_dist[k] && j != priority) * priorityQ[j][k]
//  new_priority = priority
//End
//std::tuple<Tensor<float>, Tensor<int>, int> UpdateEdges(Tensor<float> dist, Tensor<int> pQ, int priority) {
//
//}


int main(int argc, char* argv[]) {
    for (int a=0; a < N; a++) {
        v.insert({a}, a);
    }
    v.pack();

    std::cout << v << std::endl;

    for (int a=0; a < P; a++) {
        vP.insert({a}, a);
    }
    vP.pack();

    std::cout << vP << std::endl;

    edges.insert({1, 0}, 1);
    edges.insert({2, 0}, 1);
    edges.insert({0, 1}, 1);
    edges.insert({2, 1}, 1);
    edges.insert({0, 2}, 1);
    edges.pack();

    std::cout << edges << std::endl;

    weights.insert({1, 0}, 1.0);
    weights.insert({2, 0}, 3.0);
    weights.insert({0, 1}, 4.0);
    weights.insert({2, 1}, 1.0);
    weights.insert({0, 2}, 1.0);
    weights.pack();

    std::cout << weights << std::endl;

    auto out = Init();

    auto dist = std::get<0>(out);
    auto pQ = std::get<1>(out);

    std::cout << "Before new dist" << std::endl;
    auto nd = New_Dist(dist, pQ, 0);
    for (int i=0; i < N; i++) {
        std::cout << nd(i) << std::endl;
    }
    // std::cout << nd << std::endl;
}