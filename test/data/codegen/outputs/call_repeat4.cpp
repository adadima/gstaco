int f(int A) {
    int C;

    {
        auto init = A;
        C = init;
    }

return C;
}

Tensor<int, 1> func(Tensor<int, 2> B) {
    Tensor<int, 1> A({N});

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
}()) * B.at({i, j});
            }
            A.at({i}) = init_j;
        }
    }

return A;
}

