auto out = ([&]{
auto out = fib(C, D);
auto& [out0, out1] = out;
while(!(C == 0)) {
    std::tie(out0, out1) = fib(out0, out1);
}
return std::tuple<int, float>{out0, out1};
}());
A = std::get<0>(out);
B = std::get<1>(out);
