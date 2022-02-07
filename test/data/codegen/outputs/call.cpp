int A;
float B;
{
    auto init = fib(C, D);
    A = std::get<0>(init);
}
{
    auto init = fib(C, D);
    B = std::get<1>(init);
}
