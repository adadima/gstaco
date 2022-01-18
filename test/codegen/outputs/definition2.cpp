for(int i=0; i<N; i++) {
    for(int k=0; k<M; k++) {
        auto init_j = 0;
        for(int j=0; j<10; j++) {
            auto init_l = 0;
            for(int l=0; l<20; l++) {
                init_l = init_l + B[i][j][k][l] * C[j] * D[l];
            }
            init_j = init_j + init_l;
        }
        A[i][k] = init_j;
    }
}
