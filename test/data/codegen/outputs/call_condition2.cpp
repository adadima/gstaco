{
    for(int i=0; i<N; i++) {
        auto init_j = 1;
        for(int j=0; j<M; j++) {
            auto init_k = 0;
            for(int k=0; k<P; k++) {
                init_k = init_k + (([&]{
auto out = f(i);
auto& out0 = out;
                while(!(i < 99)) {
                    out0 = f(out0);
                }
return out0;
}()) + B[i][j]) * C[k];
            }
            init_j = init_j * init_k;
        }
        A[i] = init_j;
    }
}
