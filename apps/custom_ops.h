//
// Created by Alexandra Dima on 16.03.2022.
//
#include "taco/index_notation/tensor_operator.h"
#include "taco/index_notation/index_notation.h"
#include "taco/ir/ir.h"

using namespace taco;
struct Choose {

    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Add::make(
                ir::Mul::make(ir::Cast::make(ir::Eq::make(v[1], ir::Literal::zero(Float32)), Float32), v[0]),
                ir::Mul::make(ir::Cast::make(ir::Neq::make(v[1], ir::Literal::zero(Float32)), Float32), v[1])
        );
    }
};

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
        if (v.size() >= 2) {
            return ir::Min::make(v[0], v[1]);
        } else {
            return v[0];
        }
    }
};

struct CustomOrImpl {
    ir::Expr operator()(const std::vector<ir::Expr> &v) {
        return ir::Cast::make(ir::Or::make(v[0], v[1]), Int64);
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

Func customMin("Min", CustomMinImpl(), {Identity(std::numeric_limits<double>::infinity()) });

Func Or("Or", CustomOrImpl(), {Annihilator((int)1), Identity(Literal((int)0))});

Func choose("Choose", Choose(), {Annihilator((int)1), Identity(Literal((int)2))});

Func GeneralMul("Mul", CustomMulImpl(), IntersectGen(), {Annihilator(std::numeric_limits<double>::infinity()), Identity(Literal((int)0))});