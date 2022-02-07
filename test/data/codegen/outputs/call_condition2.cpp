Tensor<int, 1> A({N});
{
    for(int i=0; i<N; i++) {
        auto init_j = 1;
        for(int j=0; j<M; j++) {
            auto init_k = 0;
            for(int k=0; k<P; k++) {
                init_k = init_k + (([&]{
auto out = f(i);
auto& out0 = out;
                while(!(i < 100)) {
                    out0 = f(out0);
                }
return out0;
}()) + B.at({i, j})) * C.at({k});
            }
            init_j = init_j * init_k;
        }
        A.at({i}) = init_j;
    }
}
