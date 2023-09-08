std::tuple<int, Tensor<int, 1>, int> repeated(int a, Tensor<int, 1> b, int c) {
    int na;

    {
        auto init = a;
        na = init;
    }

    Tensor<int, 1> nb({10});

    {
        for(int i=0; i<10; i++) {
            auto init = b.at({i});
            nb.at({i}) = init;
        }
    }

    int nc;

    {
        auto init = c;
        nc = init;
    }

return std::tuple<int, Tensor<int, 1>, int>{na, nb, nc};
}

std::tuple<int, Tensor<int, 1>, int> Init(int d, Tensor<int, 1> p) {
    int nd;
    Tensor<int, 1> np({10});
    int r;

    {
        auto init = ([&]{
auto out = repeated(d, p, 0);
auto& [out0, out1, out2] = out;
        while(!(out1.at({out2}) == 0)) {
            std::tie(out0, out1, out2) = repeated(out0, out1, out2);
        }
return std::tuple<int, Tensor<int, 1>, int>{out0, out1, out2};
}());
        nd = std::get<0>(init);
    }
    {
        auto init = ([&]{
auto out = repeated(d, p, 0);
auto& [out0, out1, out2] = out;
        while(!(out1.at({out2}) == 0)) {
            std::tie(out0, out1, out2) = repeated(out0, out1, out2);
        }
return std::tuple<int, Tensor<int, 1>, int>{out0, out1, out2};
}());
        np = std::get<1>(init);
    }
    {
        auto init = ([&]{
auto out = repeated(d, p, 0);
auto& [out0, out1, out2] = out;
        while(!(out1.at({out2}) == 0)) {
            std::tie(out0, out1, out2) = repeated(out0, out1, out2);
        }
return std::tuple<int, Tensor<int, 1>, int>{out0, out1, out2};
}());
        r = std::get<2>(init);
    }

return std::tuple<int, Tensor<int, 1>, int>{nd, np, r};
}

