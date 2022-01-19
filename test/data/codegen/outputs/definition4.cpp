for(int i=0; i<10; i++) {
    auto init_j = 0;
    for(int j=0; j<20; j++) {
        init_j = init_j + B[i][j] * C[j] * C[k];
    }
    A[i] = init_j;
}
