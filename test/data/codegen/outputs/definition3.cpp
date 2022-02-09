Tensor<int, 2> edges;
Tensor<int, 1> func(Tensor<int, 2> frontier_list, Tensor<int, 1> visited, int round) {
    Tensor<int, 1> frontier({N});

    {
        for(int j=0; j<N; j++) {
            auto init_k = 0;
            for(int k=0; k<M; k++) {
                init_k = init_k || edges.at({j, k}) * frontier_list.at({round, k}) * (visited.at({j}) == 0);
            }
            frontier.at({j}) = init_k;
        }
    }

return frontier;
}

