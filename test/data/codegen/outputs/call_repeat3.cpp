Tensor<int, 1> A({N});
{
    for(int i=0; i<N; i++) {
        auto init = ([&]{
auto out = f(2);
auto& out0 = out;
        for(int iter=0; iter<5; iter++) {
            out0 = f(out0);
        }
return out0;
}()) * B.at({i});
        A.at({i}) = init;
    }
}
