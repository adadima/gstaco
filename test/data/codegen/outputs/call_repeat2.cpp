Tensor<int, 1> A({N});
Tensor<int, 1> B({N});
{
    for(int i=0; i<N; i++) {
        auto init = ([&]{
auto out = fib(1, 2);
auto& [out0, out1] = out;
        for(int iter=0; iter<2; iter++) {
            std::tie(out0, out1) = fib(out0, out1);
        }
return std::tuple<int, int>{out0, out1};
}());
        A.at({i}) = std::get<0>(init);
    }
}
{
    for(int i=0; i<N; i++) {
        auto init = ([&]{
auto out = fib(1, 2);
auto& [out0, out1] = out;
        for(int iter=0; iter<2; iter++) {
            std::tie(out0, out1) = fib(out0, out1);
        }
return std::tuple<int, int>{out0, out1};
}());
        B.at({i}) = std::get<1>(init);
    }
}
