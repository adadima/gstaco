for(int i=0; i<N; i++) {
    auto out = ([&]{
auto out = fib(1, 2);
auto& [out0, out1] = out;
    for(int iter=0; iter<2; iter++) {
        std::tie(out0, out1) = fib(out0, out1);
    }
return std::tuple<int, int>{out0, out1};
}());
    A[i] = std::get<0>(out);
    B[i] = std::get<1>(out);
}
