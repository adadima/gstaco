//
// Created by Alexandra Dima on 17.03.2022.
//

//TODO: get rid of this ASAP
#include "taco/tensor.h"
#include "custom_ops.h"
#include "taco/index_notation/kernel.h"
#include "utils.h"

int N = 1971281;

int P = N;
auto s = 0;
auto source = Literal(s);
//
Format csr({Dense,Sparse});
Tensor<int> edges("edges", {N, N}, csr);
Tensor<int> weights("weights", {N, N}, csr);
Tensor<int> v("v", {N}, Format({Dense}));
Tensor<int> vP("vP", {P}, Format({Dense}));

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
std::tuple<Tensor<int>, Tensor<int>> Init() {
    Tensor<int> dist("dist", {N}, Format({Dense}));
    IndexVar i, j;

    dist(i) = Cast(neq(v(i), source), Float64) * P;
    dist.evaluate();

    Tensor<int> pQ("priorityQ", {P, N}, Format({Dense, Sparse}), 0);
    pQ.insert({0, s}, 1);
    for (int n=0; n < N; n++) {
        if (n != s) {
                    pQ.insert({P-1, n}, 1);
        }
    }
    pQ.pack();
//    auto stmt = forall(i,
//                       forall(j,
//                              Assignment(pQ(i, j), pq_init(vP(i), 0, v(j), source, P-1))
//                              ));
//    pQ(i, j) = pq_init(i, 0, j, source, P-1);
//    pQ.evaluate();
//    Kernel k = compile(stmt);
//    k(pQ.getStorage(), vP.getStorage(), v.getStorage());
//    pQ.setStorage(pQ.getStorage());

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

Func new_dist_op("new_dist_op", newDistOp(Int64));

//  new_dist[j] = edges[j][k] * priorityQ[priority][k] * (weights[j][k] + dist[k]) + (edges[j][k] * priorityQ[priority][k] == 0) * P | k:(MIN, dist[j])

TensorStorage New_Dist(TensorStorage dist_s, TensorStorage pQ_s, int priority) {
    Tensor<int> dist("dist", {N}, Format({Dense}));
    dist.setStorage(dist_s);

    Tensor<int> pQ("priorityQ", {P, N}, Format({Dense, Sparse}));
    pQ.setStorage(pQ_s);

    Tensor<int> new_dist("new_dist", {N}, Format({Dense}));
    IndexVar j, k, p;

    auto stmt = Sequence(
            forall(j,
                   new_dist(j) = dist(j)
            ),
                    forall(p,
                            forall(j,
                                   forall(k,
                                          Assignment(
                                                  new_dist(j),
                                                  new_dist_op(
                                                          edges(j, k),
                                                          pQ(p(priority, priority + 1), k),
                                                          weights(j, k),
                                                          dist(k),
                                                          edges(j, k),
                                                          pQ(p(priority, priority + 1), k),
                                                          0,
                                                          P
                                                  ),
                                                  customMin()
                                          )
                                   )
                            )
                    )
    );
    auto kernel = compile(stmt);
//    std::cout << kernel << std::endl;
    kernel.assemble(new_dist.getStorage(), dist.getStorage(),  edges.getStorage(), pQ.getStorage(), weights.getStorage());
    kernel.compute(new_dist.getStorage(), dist.getStorage(),  edges.getStorage(), pQ.getStorage(), weights.getStorage());
    return new_dist.getStorage();
}


struct newPQOp {
    Datatype type;
    newPQOp(Datatype type) : type(type) {}

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        taco_iassert(v.size() == 8);
        return ir::Add::make(
                    ir::Mul::make(
                            ir::Cast::make(ir::Gt::make(v[0], v[1]), type),
                            ir::Cast::make(
                                    ir::And::make(
                                            ir::Lte::make(v[2], v[3]),
                                            ir::Lt::make(v[4], v[5])
                                            ),
                                            type
                                    )
                            ),
                    ir::Mul::make(
                            ir::Cast::make(
                                    ir::And::make(
                                            ir::Eq::make(v[6], v[7]),
                                            ir::Neq::make(v[8], v[9])
                                            ),
                                            type
                                    ),
                                    v[10]
                            )

        );
    }
};

Func new_pq_op("new_pq_op", newPQOp(Int64), FullSpace());


// new_priorityQ[j][k] = (dist[k] > new_dist[k]) * (j <= new_dist[k] &&  new_dist[k] < j + 1) + (dist[k] == new_dist[k] && j != priority) * priorityQ[j][k]
TensorStorage New_PQ(TensorStorage dist_s, TensorStorage new_dist_s, TensorStorage old_PQ_s, int priority) {
    Tensor<int> dist("dist", {N}, Format({Dense}));
    dist.setStorage(dist_s);

    Tensor<int> old_PQ("priorityQ", {P, N}, Format({Dense, Sparse}));
    old_PQ.setStorage(old_PQ_s);

    Tensor<int> new_dist("new_dist", {N}, Format({Dense}));
    new_dist.setStorage(new_dist_s);

    Tensor<int> new_PQ("new_priorityQ", {P, N}, Format({Dense, Sparse}));

    IndexVar j, k;
    auto stmt = forall(j,
                       forall(k,
                              Assignment(new_PQ(j, k), new_pq_op(dist(k), new_dist(k), vP(j), new_dist(k), new_dist(k), vP(j) + 1,
                                                   dist(k), new_dist(k), vP(j), priority, old_PQ(j, k)))
                       )
    );
    auto kernel = compile(stmt);
    kernel(new_PQ.getStorage(), dist.getStorage(), new_dist.getStorage(), vP.getStorage(), old_PQ.getStorage());
    return new_PQ.getStorage();
}
//Let UpdateEdges(dist float[N], priorityQ int[P][N], priority int) -> (new_dist float[N], new_priorityQ int[P][N], new_priority int)
//  new_dist[j] = edges[j][k] * priorityQ[priority][k] * (weights[j][k] + dist[k]) + (edges[j][k] * priorityQ[priority][k] == 0) * P | k:(MIN, dist[j])
//  new_priorityQ[j][k] = (dist[k] > new_dist[k]) * (j <= new_dist[k] &&  new_dist[k] < j + 1) + (dist[k] == new_dist[k] && j != priority) * priorityQ[j][k]
//  new_priority = priority
//End
std::tuple<TensorStorage, TensorStorage, int> UpdateEdges(TensorStorage dist, TensorStorage priorityQ, int priority) {
    auto new_dist = New_Dist(dist, priorityQ, priority);
    auto new_PQ = New_PQ(dist, new_dist, priorityQ, priority);
    return {new_dist, new_PQ, priority};
}

bool continue_loop(TensorStorage priorityQ_s, int priority, int target) {
    Tensor<int> priorityQ("priorityQ", {P, N}, Format({Dense, Sparse}));
    priorityQ.setStorage(priorityQ_s);

    Tensor<int> priorityQ_slice("priorityQ_slice", {N}, Format({Dense}));
    IndexVar i, j;

    priorityQ_slice(j) = priorityQ(i(priority, priority+1), j);
    priorityQ_slice.evaluate();
    return !hasFillValue(priorityQ_slice, target);
}

//  Let SSSP_one_priority_lvl(dist float[N], priorityQ int[P][N], priority int) -> (new_dist float[N], new_priorityQ int[P][N], new_priority int)
//      new_dist, new_priorityQ, _ = UpdateEdges*(dist, priorityQ, priority) | (#2[#3] == 0)
//      new_priority = priority + 1
//  End
std::tuple<TensorStorage, TensorStorage, int> SSSP_one_priority_lvl(TensorStorage dist, TensorStorage priorityQ, int priority) {
    auto new_dist = dist;
    auto new_priorityQ = priorityQ;
    auto new_priority = priority;

    //std::cout << " New priority: " << new_priority << std::endl;

    while (continue_loop(new_priorityQ, new_priority, 0)) {
        auto out = UpdateEdges(new_dist, new_priorityQ, new_priority);

        new_dist = std::get<0>(out);
        new_priorityQ = std::get<1>(out);
        new_priority = std::get<2>(out);
    }

    return {new_dist, new_priorityQ, new_priority + 1};
}

bool continue_loop2(TensorStorage newpq_s, int new_p) {
    //std::cout << "New PQ: " << newpq_s << std::endl;
    Tensor<int> priorityQ("priorityQ", {P, N}, Format({Dense, Sparse}));
    priorityQ.setStorage(newpq_s);

    return !hasFillValue(priorityQ, 0) && new_p != P;
}
//  Let SSSP() -> (new_dist float[N], dist float[N], priorityQ int[P][N])
//      dist, priorityQ = Init(source)
//      new_dist, _, _ = SSSP_one_priority_lvl*(dist, priorityQ, 0) | (#2 == 0 || #3 == P)
//  End
TensorStorage SSSP() {
    auto [dist, priorityQ] = Init();
    auto new_dist = dist.getStorage();
    auto new_priorityQ = priorityQ.getStorage();
    auto new_priority = 0;

    while (continue_loop2(new_priorityQ, new_priority)) {
        std::cout << "Priority level: " << new_priority << std::endl;
        auto out = SSSP_one_priority_lvl(new_dist, new_priorityQ, new_priority);
        new_dist = std::get<0>(out);
        new_priorityQ = std::get<1>(out);
        new_priority = std::get<2>(out);
    }

    return new_dist;
}

int main(int argc, char* argv[]) {
    for (int a=0; a < N; a++) {
        v.insert({a}, a);
    }
    v.pack();

    for (int a=0; a < P; a++) {
        vP.insert({a}, a);
    }
    vP.pack();
    auto filename = get_test_dir() + "codegen/graphs/graph-dataset-small/roadNet-CA.weighted.mtx";
    auto tensors = loadEdgesFromFile(filename);

    std::cout << "Loaded edges and weights" << std::endl;
    N = std::get<0>(tensors);
    edges = std::get<1>(tensors);
    weights = std::get<2>(tensors);

//    edges.insert({1, 0}, 1);
//    edges.insert({2, 0}, 1);
//    edges.insert({0, 1}, 1);
//    edges.insert({2, 1}, 1);
//    edges.insert({0, 2}, 1);
//    edges.pack();
//
//    weights.insert({1, 0}, 1);
//    weights.insert({2, 0}, 3);
//    weights.insert({0, 1}, 4);
//    weights.insert({2, 1}, 1);
//    weights.insert({0, 2}, 1);
//    weights.pack();

//    edges.insert({0, 1}, 1);
//    edges.insert({1, 2}, 1);
//    edges.insert({2, 3}, 1);
//    edges.insert({3, 4}, 1);
//    edges.insert({0, 4}, 1);
//    edges.insert({2, 4}, 1);
//    edges.insert({1, 4}, 1);
    edges.pack();

//    weights.insert({0, 1}, 1);
//    weights.insert({1, 2}, 1);
//    weights.insert({2, 3}, 1);
//    weights.insert({3, 4}, 1);
//    weights.insert({0, 4}, 10);
//    weights.insert({2, 4}, 8);
//    weights.insert({1, 4}, 6);
    weights.pack();

//    edges.insert({1, 0}, 1);
//    edges.insert({2, 0}, 1);
//    edges.insert({2, 1}, 1);
//    edges.insert({2, 3}, 1);
//    edges.pack();
//
//    weights.insert({1, 0}, 1);
//    weights.insert({2, 0}, 3);
//    weights.insert({2, 1}, 1);
//    weights.insert({2, 3}, 5);
//    weights.pack();

    auto nd = SSSP();
    std::cout << nd << std::endl;
}