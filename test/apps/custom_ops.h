//
// Created by Alexandra Dima on 16.03.2022.
//
#include "taco/index_notation/tensor_operator.h"
#include "taco/index_notation/index_notation.h"
#include "taco/ir/ir.h"

using namespace taco;
//struct Choose {
//
//    ir::Expr operator()(const std::vector<ir::Expr> &v) {
//        return ir::Add::make(
//                ir::Mul::make(ir::Cast::make(ir::Eq::make(v[1], ir::Literal::zero(Float32)), Float32), v[0]),
//                ir::Mul::make(ir::Cast::make(ir::Neq::make(v[1], ir::Literal::zero(Float32)), Float32), v[1])
//        );
//    }
//};

using namespace taco;
struct ConditionalOp {
    // value, target, tExpr, fExpr
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Add::make(
                ir::Mul::make(ir::Cast::make(ir::Eq::make(v[0], v[1]), Float32), v[2]),
                ir::Mul::make(ir::Cast::make(ir::Neq::make(v[0], v[1]), Float32), v[3])
        );
    }
};

struct UnionIntersect {
    IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
        if (regions.size() < 4) {
            return IterationAlgebra();
        }

        IterationAlgebra intersect1 = Intersect(regions[0], regions[1]);
        IterationAlgebra intersect2 = Intersect(regions[2], regions[3]);

        return Union(intersect1, intersect2);
    }
};

Func conditional("CondOp", ConditionalOp(), UnionIntersect());

struct CustomMinImpl {
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Min::make(v[0], v[1]);
    }
};

struct CustomOrImpl {
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Cast::make(ir::Or::make(v[0], v[1]), Int64);
    }
};

struct CustomAndImpl {
    Datatype type_;
    CustomAndImpl(Datatype type_) : type_(type_) {}

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Cast::make(ir::And::make(v[0], v[1]), Int64);
    }
};

struct CustomMulImpl {
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        ir::Expr e = v[0];
        for(int i=1; i < v.size(); i++) {
            e = ir::Mul::make(e, v[i]);
        }
        return e;
    }
};


struct IntersectGen {
    IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
        if (regions.size() < 2) {
            return IterationAlgebra();
        }

        IterationAlgebra intersections = regions[0];
        for(size_t i = 1; i < regions.size(); ++i) {
            intersections = Intersect(intersections, regions[i]);
        }
        return intersections;
    }
};

struct FullSpace {
    IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
        IterationAlgebra unions = regions[0];

        for(size_t i = 1; i < regions.size(); i++) {
            unions = Union(unions, regions[i]);
        }
        return unions;
    }
};

Func customMin("Min", CustomMinImpl(), {Identity(std::numeric_limits<double>::infinity()) });

Func Or("Or", CustomOrImpl(), {Annihilator((int)1), Identity(Literal((int)0))});

struct Choose {
    Datatype type;
    Choose(Datatype type) : type(type) {}

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Add::make(
                ir::Mul::make(
                        ir::Cast::make(
                                ir::Eq::make(v[1], ir::Literal::zero(type)),
                                type
                                ),
                        v[0]
                        ),
                ir::Mul::make(
                        ir::Cast::make(
                                ir::Neq::make(v[1], ir::Literal::zero(type)),
                                type
                                ),
                        v[1]
                        )
        );
    }
};

struct CustomCastImpl {
    Datatype type;
    CustomCastImpl(Datatype type) : type(type) {}

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Cast::make(v[0], type);
    };
};

Func make_choose(Datatype type) {
    Func choose("Choose", Choose(type), {Annihilator((int)1), Identity(Literal((int)2))});
    return choose;
};

Func GeneralMul("Mul", CustomMulImpl(), IntersectGen(), {Annihilator(std::numeric_limits<double>::infinity()), Identity(Literal((int)0))});

Func AndFloat("And", CustomAndImpl(Float32), {Annihilator((int)0), Identity(Literal((int)1))});

Func AndInt("And", CustomAndImpl(Int64), {Annihilator((int)0), Identity(Literal((int)1))});

Func CastInt("Cast", CustomCastImpl(Int64));

template <typename T>
bool hasFillValue(const Tensor<T>& lhs, const T& rhs) {
    for (auto &elem : lhs) {
        if (elem.second != rhs) {
            return false;
        }
    }
    return true;
}