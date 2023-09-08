Tensor<float, 2> func(Tensor<float, 4> B, Tensor<float, 1> C, Tensor<float, 1> D) {
    Tensor<float, 2> A({N, M});

    {
        for(int i=0; i<N; i++) {
            for(int k=0; k<M; k++) {
                auto init_j = 0;
                for(int j=0; j<10; j++) {
                    auto init_l = 0;
                    for(int l=0; l<20; l++) {
                        init_l = init_l + B.at({i, j, k, l}) * C.at({j}) * D.at({l});
                    }
                    init_j = init_j + init_l;
                }
                A.at({i, k}) = init_j;
            }
        }
    }

return A;
}

