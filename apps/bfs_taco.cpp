//
// Created by Alexandra Dima on 02.03.2022.
//
#include "taco/tensor.h"
#include "custom_ops.h"
#include "taco/index_notation/kernel.h"

using namespace taco;

int N = 5;
auto source = Literal(4);
//
Format csr({Dense,Sparse});
Tensor<int> edges("edges", {N, N}, csr);
Tensor<int> v("v", {N}, Format({Dense}));


std::tuple<Tensor<int>, Tensor<int>> Init() {
    // F[j] = (j == source)
    Tensor<int> F("F", {N}, Format({Dense}));
    IndexVar i;

    F(i) = eq(v(i), source);
    F.evaluate();
    std::cout << F << std::endl;

    // P[j] = (j == source) * (0 - 2) + (j != source) * (0 - 1)
    Tensor<int> P("P", {N}, Format({Dense}));
    P(i) = conditional(v(i), source, Literal(0-2), Literal(0-1));   //eq(v(i), source) * Literal(0 - 2) + neq(v(i), source) * Literal(0 - 1);
    P.evaluate();
    std::cout << P << std::endl;

    return {F, P};
}

//  V_out[j] = P_in[j] == 0 - 1
Tensor<int> get_V(Tensor<int> P_in) {
    IndexVar i;
    Tensor<int> V_out("V_out", {N}, Format({Dense}));

    V_out(i) = eq(P_in(i), 0-1);
    V_out.evaluate();
    return V_out;
}

// F_out[j] = edges[j][k] * F_in[k] * V_out[j] | k: (OR, 0)   / k:(OR, 0)
Tensor<int> get_F(Tensor<int> F_in, Tensor<int> V_out) {
    Tensor<int> F_out("F_out", {N}, Format({Dense}), 0);
    std::cout << "F in storage before: " << F_in.getStorage() << std::endl;
    IndexVar i, j;
    auto stmt = forall(i,
                        forall(j,
                            Assignment(
                                    F_out(i),
                                    GeneralMul(edges(i, j), F_in(j), V_out(i)),
                                    Or())
                       ));
    Kernel k = compile(stmt);
    k(F_out.getStorage(), edges.getStorage(), F_in.getStorage(), V_out.getStorage());
    F_out.setStorage(F_out.getStorage());
    return F_out;
}

//  P_out[j] = edges[j][k] * F_in[k] * V_out[j] * (k + 1) | k:(CHOOSE, P_in[j])
Tensor<int> get_P(Tensor<int> F_in, Tensor<int> V_out, Tensor<int> P_in) {
    // TensorStorage s = TensorStorage(Int64, {N}, Format({Dense}), 0);
    Tensor<int> P_out("P_out", {N}, Format({Dense}));

    IndexVar i;
    IndexVar j;
    auto stmt = Sequence(
                forall(i,
                       Assignment(P_out(i), P_in(i))
                ),
                forall(i,
                           forall(j,
                                  Assignment(
                                          P_out(i),
                                          GeneralMul(edges(i, j), F_in(j), V_out(i), v(j) + 1),
                                          make_choose(Float32)())
                                  )
                              )
                );
    Kernel k = compile(stmt);
    k.assemble(P_out.getStorage(), P_in.getStorage(), edges.getStorage(), F_in.getStorage(), V_out.getStorage(), v.getStorage());
    k.compute(P_out.getStorage(), P_in.getStorage(), edges.getStorage(), F_in.getStorage(), V_out.getStorage(), v.getStorage());
    P_out.setStorage(P_out.getStorage());
    return P_out;
}

//Let BFS_Step(F_in int[N], P_in int[N], V_in int[N]) -> (F_out int[N], P_out int[N], V_out int[N])
//  V_out[j] = P_in[j] == 0 - 1
//  F_out[j] = edges[j][k] * F_in[k] * V_out[j] | k: (OR, 0)   / k:(CHOOSE, 0)
//  P_out[j] = edges[j][k] * F_in[k] * V_out[j] * (k + 1) | k:(CHOOSE, P_in[j])
//End
std::tuple<Tensor<int>, Tensor<int>, Tensor<int>> BFS_Step(Tensor<int> F_in, Tensor<int> P_in, Tensor<int> V_in) {

    auto V_out = get_V(P_in);
    std::cout << V_out << std::endl;

    auto F_out = get_F(F_in, V_out);
    std::cout << F_out.getStorage() << std::endl;

    auto P_out = get_P(F_in, V_out, P_in);
    std::cout << P_out << std::endl;

    return {F_out, P_out, V_out};
}


//Let BFS() -> (P_out int[N], F_in int[N], P_in int[N], V int[N])
//  F_in, P_in = Init()
//  V[i] = 0
//  _, P_out, _ = BFS_Step*(F_in, P_in, V) | (#1 == 0)
//End
std::tuple<Tensor<int>, Tensor<int>, Tensor<int>, Tensor<int>> BFS() {
    auto out = Init();
    auto F = std::get<0>(out);
    auto P = std::get<1>(out);

    Tensor<int> V("V", {N}, Format({Dense}));
    IndexVar i;
    V(i) = IndexExpr(0);

    Tensor<int> F_in("F_in", {N}, Format({Dense}));
    F_in.setStorage(F.getStorage());

    Tensor<int> P_in("P_in", {N}, Format({Dense}));
    P_in.setStorage(P.getStorage());

    Tensor<int> V_in("V_in", {N}, Format({Dense}));
    V_in.setStorage(V.getStorage());

    while (!hasFillValue(F, 0)) {
        std::cout << "Current frontier: " << F << std::endl;

        auto out = BFS_Step(F_in, P_in, V_in);
        F = std::get<0>(out);
        F_in.setStorage(F.getStorage());

        P = std::get<1>(out);
        P_in.setStorage(P.getStorage());

        V = std::get<2>(out);
        V_in.setStorage(V.getStorage());
    }

    return {P, F, P, V};
}


int main(int argc, char* argv[]) {
    for (int a=0; a < N; a++) {
        v.insert({a}, a);
    }
    v.pack();

//    edges.insert({1, 0}, 1);
//    edges.insert({2, 0}, 1);
//    edges.insert({2, 1}, 1);
//    edges.insert({0, 2}, 1);
//    edges.insert({2, 3}, 1);

//    edges.insert({1, 0}, 1);
//    edges.insert({2, 0}, 1);
//    edges.insert({0, 1}, 1);
//    edges.insert({2, 1}, 1);
//    edges.insert({0, 2}, 1);

    edges.insert({0, 1}, 1);
    edges.insert({1, 2}, 1);
    edges.insert({2, 3}, 1);
    edges.insert({3, 4}, 1);
    edges.insert({0, 4}, 1);
    edges.pack();

    std::cout << "Edges: " << edges << std::endl;
    auto [P, F, _, V] = BFS();

    std::cout << "P out: " << P << std::endl;
}