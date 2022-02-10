std::tuple<int, int, int> repeated(int a, int b, int c) {
    int na;

    {
        auto init = a;
        na = init;
    }

    int nb;

    {
        auto init = b;
        nb = init;
    }

    int nc;

    {
        auto init = c;
        nc = init;
    }

return std::tuple<int, int, int>{na, nb, nc};
}

std::tuple<int, int, int> Init(int d, int p) {
    int nd;
    int np;
    int r;

    {
        auto init = ([&]{
auto out = repeated(d, p, 0);
auto& [out0, out1, out2] = out;
        while(!(out1 == 0)) {
            std::tie(out0, out1, out2) = repeated(out0, out1, out2);
        }
return std::tuple<int, int, int>{out0, out1, out2};
}());
        nd = std::get<0>(init);
    }
    {
        auto init = ([&]{
auto out = repeated(d, p, 0);
auto& [out0, out1, out2] = out;
        while(!(out1 == 0)) {
            std::tie(out0, out1, out2) = repeated(out0, out1, out2);
        }
return std::tuple<int, int, int>{out0, out1, out2};
}());
        np = std::get<1>(init);
    }
    {
        auto init = ([&]{
auto out = repeated(d, p, 0);
auto& [out0, out1, out2] = out;
        while(!(out1 == 0)) {
            std::tie(out0, out1, out2) = repeated(out0, out1, out2);
        }
return std::tuple<int, int, int>{out0, out1, out2};
}());
        r = std::get<2>(init);
    }

return std::tuple<int, int, int>{nd, np, r};
}

