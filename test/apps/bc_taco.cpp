//
// Created by Alexandra Dima on 28.03.2022.
//
#include "taco/tensor.h"
#include "custom_ops.h"
#include "taco/index_notation/kernel.h"

using namespace taco;

int N = 5;
Literal source = Literal(4);
//
Format csr({Dense,Sparse});
Tensor<int> edges("edges", {N, N}, csr);
Tensor<int> v("v", {N}, Format({Dense}));

struct Op1 {
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Cast::make(
                ir::And::make(
                        ir::Eq::make(v[0], v[1]),
                        ir::Eq::make(v[2], v[3])
                ),
                Int64
        );
    }
};

Func op1("op1", Op1());

//  Let Init() -> (frontier_list int[N][N], num_paths int[N], deps int[N], visited int[N])
//      num_paths[j] = (j == source)
//      deps[j] = 0
//      visited[j] = (j == source)
//      frontier_list[r][j] = (r == 0 && j == source)
//  End

std::tuple<Tensor<int>, Tensor<int>, Tensor<int>, Tensor<int>> Init() {
    Tensor<int> num_paths("num_paths", {N}, Format({Dense}));
    {
        IndexVar j;
        num_paths(j) = eq(v(j), source);
    }

    Tensor<int> deps("deps", {N}, Format({Dense}));
    {
        IndexVar j;
        deps(j) = Literal::zero(Int64);
    }

    Tensor<int> visited("visited", {N}, Format({Dense}));
    {
        IndexVar j;
        visited(j) = eq(v(j), source);
    }

    Tensor<int> frontier_list("frontier_list", {N, N}, Format({Dense, Dense}));
    {
        IndexVar r, j;
        frontier_list(r, j) = op1(v(r), 0, v(j), source);
    }

    std::cout << frontier_list << std::endl;
    std::cout << num_paths << std::endl;
    std::cout << deps << std::endl;
    std::cout << visited << std::endl;
    return {frontier_list, num_paths, deps, visited};
}

struct Op2 {
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Mul::make(
                    v[0],
                    ir::Mul::make(
                            v[1],
                            ir::Cast::make(
                                    ir::Eq::make(
                                            v[2],
                                            v[3]
                                            ),
                                    Int64
                                    )
                            )
                );
    }
};

Func op2("op2", Op2());

struct Op3 {
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Add::make(
                ir::Mul::make(
                        v[0],
                        ir::Cast::make(
                                ir::Eq::make(
                                        v[1], v[2]
                                        ),
                                        Int64
                                )
                        ),
                ir::Mul::make(
                        v[3],
                        ir::Cast::make(
                                ir::Neq::make(
                                        v[4],
                                        v[5]
                                ),
                                Int64
                        )
                )
        );
    }
};

Func op3("op3", Op3());

struct Op4 {
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Mul::make(
                v[0],
                ir::Mul::make(
                        v[1],
                        ir::Mul::make(
                                ir::Cast::make(
                                        ir::Eq::make(
                                                v[2],
                                                v[3]
                                        ),
                                        Int64
                                ),
                                v[4]
                                )

                )
        );
    }
};

Func op4("op4", Op4());

struct Op5 {
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Mul::make(
                v[0],
                ir::Mul::make(
                        v[1],
                        ir::Cast::make(
                                        ir::Eq::make(
                                                v[2],
                                                v[3]
                                        ),
                                        Int64
                                )
                )
        );
    }
};

Func op5("op5", Op5(), UnionIntersect());

// frontier[j] = edges[j][k] * frontier_list[round][k] * (visited[j] == 0) | k:(CHOOSE, 0)
Tensor<int> Frontier(Tensor<int> frontier_list, Tensor<int> visited, int round) {
    Tensor<int> frontier("frontier", {N}, Format({Dense}));
    {
        IndexVar i, j, k, r;
        // frontier(j) = op2(...) |
        auto stmt = Sequence(
                forall(j,
                       Assignment(frontier(j), Literal::zero(Int64))
                ),
                forall(i,
                       forall(j,
                              forall(k,
                                     Assignment(
                                             frontier(j),
                                             op2(edges(j, k), frontier_list(i(round-1, round), k), visited(j), 0),
                                             make_choose(Int64)())
                                     )
                       )
                )
        );
        Kernel kernel = compile(stmt);
        kernel.assemble(frontier.getStorage(), edges.getStorage(), frontier_list.getStorage(), visited.getStorage());
        kernel.compute(frontier.getStorage(), edges.getStorage(), frontier_list.getStorage(), visited.getStorage());
        frontier.setStorage(frontier.getStorage());
        std::cout << frontier << std::endl;
    }

    return frontier;
}

//forward_frontier_list[r][j] = frontier[j] * (r == round) + frontier_list[r][j] * (r != round)
Tensor<int> Forward_Frontier_List(Tensor<int> frontier, Tensor<int> frontier_list, int round) {
    Tensor<int> forward_flist("forward_frontier_list", {N, N}, Format({Dense, Dense}));
    {
        IndexVar r, j;
        auto stmt = forall(r,
                           forall(j,
                                  Assignment(forward_flist(r, j), op3(frontier(j), v(r), round, frontier_list(r, j), v(r), round))
                                )
                           );
        Kernel kernel = compile(stmt);
        kernel.assemble(forward_flist.getStorage(), frontier.getStorage(), v.getStorage(), frontier_list.getStorage());
        kernel.compute(forward_flist.getStorage(), frontier.getStorage(), v.getStorage(), frontier_list.getStorage());
        forward_flist.setStorage(forward_flist.getStorage());
        std::cout << forward_flist << std::endl;
    }
    return forward_flist;
}

// forward_num_paths[j] = edges[j][k] * frontier_list[round - 1][k] * (visited[j] == 0) * num_paths[k] | k:(+, num_paths[j])
Tensor<int> Forward_Num_Paths(Tensor<int> frontier_list, Tensor<int> visited, Tensor<int> num_paths, int round) {
    Tensor<int> forward_np("forward_num_paths", {N}, Format({Dense}));
    {
        IndexVar i, j, k;
        auto stmt = Sequence(
                forall(j,
                       Assignment(forward_np(j), num_paths(j))
                       ),
                       forall(i,
                              forall(j,
                                     forall(k,
                                            Assignment(forward_np(j),
                                                       op4(edges(j, k), frontier_list(i(round-1, round), k), visited(j), 0, num_paths(k)),
                                                       Add())
                                            )))
                );
        Kernel kernel = compile(stmt);
        kernel.assemble(forward_np.getStorage(), num_paths.getStorage(), edges.getStorage(), frontier_list.getStorage(), visited.getStorage());
        kernel.compute(forward_np.getStorage(), num_paths.getStorage(), edges.getStorage(), frontier_list.getStorage(), visited.getStorage());
        forward_np.setStorage(forward_np.getStorage());
        std::cout << forward_np << std::endl;
    }
    return forward_np;
}

//      forward_visited[j] = edges[j][k] * frontier_list[round-1][k] * (visited[j] == 0) | k:(CHOOSE, visited[j])
Tensor<int> Forward_Visited(Tensor<int> frontier_list, Tensor<int> visited, int round) {
    Tensor<int> forward_v("forward_visited", {N}, Format({Dense}));
    {
        IndexVar i, j, k;
        auto stmt = Sequence(
                forall(j,
                       Assignment(forward_v(j), visited(j))
                ),
                forall(i,
                       forall(j,
                              forall(k,
                                     Assignment(forward_v(j),
                                                op5(edges(j, k), frontier_list(i(round-1, round), k), visited(j), 0),
                                                make_choose(Int64)())
                              )))
        );
        Kernel kernel = compile(stmt);
        kernel.assemble(forward_v.getStorage(), visited.getStorage(), edges.getStorage(), frontier_list.getStorage());
        kernel.compute(forward_v.getStorage(), visited.getStorage(), edges.getStorage(), frontier_list.getStorage());
        forward_v.setStorage(forward_v.getStorage());
        std::cout << forward_v << std::endl;
    }
    return forward_v;
}
//  Let Forward_Step(frontier_in int[N], frontier_list int[N][N], num_paths int[N], visited int[N], round int) -> (frontier int[N], forward_frontier_list int[N][N], forward_num_paths int[N], forward_visited int[N], forward_round int)
//      frontier[j] = edges[j][k] * frontier_list[round - 1][k] * (visited[j] == 0) | k:(CHOOSE, 0)
//      forward_frontier_list[r][j] = frontier[j] * (r == round) + frontier_list[r][j] * (r != round)
//      forward_num_paths[j] = edges[j][k] * frontier_list[round - 1][k] * (visited[j] == 0) * num_paths[k] | k:(+, num_paths[j])
//      forward_visited[j] = edges[j][k] * frontier_list[round-1][k] * (visited[j] == 0) | k:(CHOOSE, visited[j])
//      forward_round = round + 1
//  End
std::tuple<Tensor<int>, Tensor<int>, Tensor<int>, Tensor<int>, int> Forward_Step(Tensor<int> frontier_in, Tensor<int> frontier_list, Tensor<int> num_paths, Tensor<int> visited, int round) {
    auto f = Frontier(frontier_list, visited, round);
    auto ff = Forward_Frontier_List(f, frontier_list, round);
    auto np = Forward_Num_Paths(ff, visited, num_paths, round);
    auto fv = Forward_Visited(frontier_list, visited, round);

    return {f, ff, np, fv, round + 1};
}

//  Let Forward(frontier_list int[N][N], num_paths int[N], visited int[N]) -> (dummy int[N], new_forward_frontier_list int[N][N], new_forward_num_paths int[N], new_forward_visited int[N], new_forward_round int)
//      dummy[j] = 0
//      _, new_forward_frontier_list, new_forward_num_paths, new_forward_visited, new_forward_round = Forward_Step*(dummy, frontier_list, num_paths, visited, 1) | (#2[#5-1] == 0)
//  End
std::tuple<Tensor<int>, Tensor<int>, Tensor<int>, Tensor<int>, int> Forward(Tensor<int> frontier_list, Tensor<int> num_paths, Tensor<int> visited) {
    Tensor<int> dummy("dummy", {N}, Format({Dense}));

    {
        IndexVar j;
        dummy(j) = Literal::zero(Int64);
    }

    auto a1 = dummy;
    auto a2 = frontier_list;
    auto a3 = num_paths;
    auto a4 = visited;
    auto a5 = 1;

    IndexVar i, j;
    Tensor<int> frontier_slice("frontier_slice", {N}, Format({Dense}));
    frontier_slice(j) = a2(i(a5-1, a5), j);
    frontier_slice.evaluate();
    std::cout << frontier_slice << std::endl;

    while (!hasFillValue(frontier_slice, 0)) {
        auto out = Forward_Step(a1, a2, a3,a4, a5);
        a1.setStorage(std::get<0>(out).getStorage());
        a2.setStorage(std::get<1>(out).getStorage());
        a3.setStorage(std::get<2>(out).getStorage());
        a4.setStorage(std::get<3>(out).getStorage());
        a5 = std::get<4>(out);

        IndexVar i, j;
        Tensor<int> frontier_slice_("frontier_slice_", {N}, Format({Dense}));
        frontier_slice_(j) = a2(i(a5-1, a5), j);
        frontier_slice_.evaluate();
        frontier_slice.setStorage(frontier_slice_.getStorage());
        std::cout << frontier_slice << std::endl;
    }

    return {a1, a2, a3, a4, a5};
}

//  Let BC_Froward() -> (frontier_list int[N][N], num_paths int[N], deps int[N], visited int[N], forward_frontier_list int[N][N], forward_num_paths int[N], int forward_round)
//      frontier_list, num_paths, deps, visited = Init()
//      _, forward_frontier_list, forward_num_paths, _, forward_round = Forward(frontier_list, num_paths, visited)
//  End

std::tuple<Tensor<int>, Tensor<int>, int> BC_Forward() {
   auto [frontier_list, num_paths, deps, visited] = Init();
   auto [a1, forward_fl, forward_np, a4, forward_r] = Forward(frontier_list, num_paths, visited);
   return {forward_fl, forward_np, forward_r};
}


int main(int argc, char* argv[]) {
    for (int a=0; a < N; a++) {
        v.insert({a}, a);
    }
    v.pack();

    // change N= 3 and source =Literal(0)
//    edges.insert({1, 0}, 1);
//    edges.insert({2, 0}, 1);
//    edges.insert({2, 1}, 1);
//    edges.insert({0, 2}, 1);
//    edges.insert({2, 3}, 1);

// change N=4 and source =Literal(0)
//    edges.insert({1, 0}, 1);
//    edges.insert({2, 0}, 1);
//    edges.insert({0, 1}, 1);
//    edges.insert({2, 1}, 1);
//    edges.insert({0, 2}, 1);

// change N=5 and source =Literal(4)
    edges.insert({0, 1}, 1);
    edges.insert({1, 2}, 1);
    edges.insert({2, 3}, 1);
    edges.insert({3, 4}, 1);
    edges.insert({0, 4}, 1);
    edges.pack();

    std::cout << "Edges: " << edges << std::endl;
//    auto [frontier_list, num_paths, deps, visited] = Init();

//    auto f = Frontier(frontier_list, visited, 1);
//    auto ff = Forward_Frontier_List(f, frontier_list, 1);
//    auto np = Forward_Num_Paths(ff, visited, num_paths, 1);
//    auto fv = Forward_Visited(frontier_list, visited, 1);
    auto [frontier, paths, round] = BC_Forward();

    std::cout << frontier << std::endl;
    std::cout << paths << std::endl;
    std::cout << round << std::endl;
}