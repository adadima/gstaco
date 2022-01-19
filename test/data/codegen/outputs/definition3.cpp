for(int j=0; j<N; j++) {
    auto init_k = 0;
    for(int k=0; k<M; k++) {
        init_k = init_k || edges[j][k] * frontier_list[round][k] * (visited[j] == 0);
    }
    frontier[j] = init_k;
}
