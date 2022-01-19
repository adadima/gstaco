{
    for(int i=0; i<N; i++) {
        auto init_j = 1;
        for(int j=0; j<M; j++) {
            init_j = init_j * ([&]{
auto out = f(j);
auto& out0 = out;
            for(int iter=0; iter<5; iter++) {
                out0 = f(out0);
            }
return out0;
}()) * B[i][j];
        }
        A[i] = init_j;
    }
}
