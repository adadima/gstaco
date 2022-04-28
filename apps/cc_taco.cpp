//
// Created by Alexandra Dima on 30.03.2022.
//
#include "taco/tensor.h"
#include "custom_ops.h"
#include "taco/index_notation/kernel.h"

using namespace taco;

int N = 8;

Format csr({Dense,Sparse});
Tensor<int> edges("edges", {N, N}, csr);
Tensor<int> v("v", {N}, Format({Dense}));

//  Let Init() -> (IDs int[N], update int)
//      IDs[i] = i
//      update = 1
//  End
std::tuple<TensorStorage, TensorStorage> Init() {
    Tensor<int> IDs("IDs", {N}, Format({Dense}));

    IndexVar i;
    IDs(i) = v(i);
    IDs.evaluate();

    Tensor<int> update("update");
    update() = Literal(1);
    update.evaluate();

    return {IDs.getStorage(), update.getStorage()};
}

struct ForwardOp {
    Datatype type_;
    ForwardOp(Datatype type_) : type_(type_) {}

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        taco_iassert(v.size() == 8);
        return ir::Add::make(
                ir::Mul::make(v[0], v[1]),
                ir::Mul::make(
                        ir::Sub::make(1, v[2]),
                        v[3]
                        )
                );
    }
};

Func forward_op("forward_op", ForwardOp(Int64));

//  Let Forward(old_ids int[N]) -> (new_ids int[N])
//      new_ids[i] = edges[j][i] * old_ids[j] + (1 - edges[j][i]) * old_ids[i] | j : (MIN, old_ids[i])
//  End
TensorStorage Forward(TensorStorage old_ids_s) {
    Tensor<int> old_ids("old_ids", {N}, Format({Dense}));
    old_ids.setStorage(old_ids_s);

    Tensor<int> forward_ids("forward_ids", {N}, Format({Dense}));

    IndexVar i, j, k;
    auto stmt = Sequence(
            forall(k,
                   forward_ids(k) = old_ids(k)
                   ),
                   forall(j,
                          forall(i,
                                 Assignment(
                                            forward_ids(i),
                                            forward_op(edges(j, i), old_ids(j), edges(j, i), old_ids(i)),
                                            customMin()
                                            )
                                 )
                          )
            );
    auto kernel = compile(stmt);
    kernel.assemble(forward_ids.getStorage(), old_ids.getStorage(), edges.getStorage());
    kernel.compute(forward_ids.getStorage(), old_ids.getStorage(), edges.getStorage());
    return forward_ids.getStorage();
}

//  Let Backward(old_ids int[N]) -> (new_ids int[N])
//      new_ids[j] = edges[j][i] * old_ids[i] + (1 - edges[j][i]) * old_ids[j] | j : (MIN, old_ids[j])
//  End
TensorStorage Backward(TensorStorage old_ids_s) {
    Tensor<int> old_ids("old_ids", {N}, Format({Dense}));
    old_ids.setStorage(old_ids_s);

    Tensor<int> new_ids("new_ids", {N}, Format({Dense}));

    IndexVar i, j, k;
    auto stmt = Sequence(
            forall(k,
                   new_ids(k) = old_ids(k)
            ),
            forall(j,
                   forall(i,
                          Assignment(
                                  new_ids(j),
                                  forward_op(edges(j, i), old_ids(i), edges(j, i), old_ids(j)),
                                  customMin()
                          )
                   )
            )
    );
    auto kernel = compile(stmt);
    kernel.assemble(new_ids.getStorage(), old_ids.getStorage(), edges.getStorage());
    kernel.compute(new_ids.getStorage(), old_ids.getStorage(), edges.getStorage());
    return new_ids.getStorage();
}


struct Custom {
    Datatype type_;
    Custom(Datatype type_) : type_(type_) {}

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Neq::make(v[0], v[1]);
    }
};

Func custom_op("custom_op", Custom(Int64));
//  Let UpdateEdges(old_forward_ids int[N], old_ids int[N], old_update int) -> (forward_ids int[N], new_ids int[N], new_update int)
//      forward_ids = Forward(old_ids)
//      new_ids = Backward(forward_ids)
//
//      new_update = (old_ids[i] != new_ids[i]) | i : (OR, 0)
//  End
std::tuple<TensorStorage, TensorStorage, TensorStorage> UpdateEdges(TensorStorage old_forward_ids_S, TensorStorage old_ids_s, TensorStorage old_update_s) {
    Tensor<int> old_ids("old_ids", {N}, Format({Dense}));
    old_ids.setStorage(old_ids_s);

    auto forward_ids = Forward(old_ids_s);
    auto new_ids_s = Backward(forward_ids);

    Tensor<int> new_ids("new_ids", {N}, Format({Dense}));
    new_ids.setStorage(new_ids_s);

    Tensor<int> new_update("new_update");
    IndexVar i;
    auto stmt = forall(i,
                       Assignment(new_update(), custom_op(old_ids(i), new_ids(i)), Or())
                       );
    auto kernel = compile(stmt);
    kernel.assemble(new_update.getStorage(), old_ids.getStorage(), new_ids_s);
    kernel.compute(new_update.getStorage(), old_ids.getStorage(), new_ids_s);
    return {forward_ids, new_ids_s, new_update.getStorage()};
}

bool continue_loop(TensorStorage update, int fill) {
    Tensor<int> update_t("update_t");
    update_t.setStorage(update);

    return !hasFillValue(update_t, fill);
}


//Let CC() -> (ids int[N], update int, dummy int[N], new_ids int[N], new_update int)
//  ids, update = Init()
//  dummy[i] = 0
//
//  _, new_ids, new_update = UpdateEdges*(dummy, ids, update) | (#3 == 0)
//End
TensorStorage CC() {
    auto [ids, update] = Init();

    Tensor<int> dummy_t("dummy_t", {N}, Format({Dense}));
    IndexVar i;
    dummy_t(i) = Literal::zero(Int64);

    auto dummy = dummy_t.getStorage();
    while (continue_loop(update, 0)) {
        auto out = UpdateEdges(dummy, ids, update);
        dummy = std::get<0>(out);
        ids = std::get<1>(out);
        update = std::get<2>(out);
    }

    return {ids};
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
    edges.insert({6, 5}, 1);
    edges.pack();

    auto IDs = CC();

    std::cout << "IDs: " << IDs << std::endl;
}