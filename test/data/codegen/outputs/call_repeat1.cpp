int A;
float B;
{
    auto init = ([&]{
auto out = fib(C, D);
auto& [out0, out1] = out;
    for(int iter=0; iter<99; iter++) {
        std::tie(out0, out1) = fib(out0, out1);
    }
return std::tuple<int, float>{out0, out1};
}());
    A = std::get<0>(init);
}
{
    auto init = ([&]{
auto out = fib(C, D);
auto& [out0, out1] = out;
    for(int iter=0; iter<99; iter++) {
        std::tie(out0, out1) = fib(out0, out1);
    }
return std::tuple<int, float>{out0, out1};
}());
    B = std::get<1>(init);
}
