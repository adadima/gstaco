{
    for(int j=0; j<N; j++) {
        auto init_k = 0;
        for(int k=0; k<N; k++) {
            init_k = init_k || G[j][k] * F_in[k] * (P_in[j] != 0 - 1);
        }
        F_out[j] = init_k;
    }
}
{
    for(int j=0; j<N; j++) {
        auto init_k = P_in[j];
        for(int k=0; k<N; k++) {
            init_k = G[j][k] * F_in[k] * (P_in[j] != 0 - 1) * k ? G[j][k] * F_in[k] * (P_in[j] != 0 - 1) * k : init_k;
        }
        P_out[j] = init_k;
    }
}
