std::tuple<int, float> fib(int A, float B) {
    int C;

    {
        auto init = 1;
        C = init;
    }

    float D;

    {
        auto init = 2;
        D = init;
    }

return std::tuple<int, float>{C, D};
}

std::tuple<int, float> func(int C, float D) {
    int A;
    float B;

    {
        auto init = fib(C, D);
        A = std::get<0>(init);
    }
    {
        auto init = fib(C, D);
        B = std::get<1>(init);
    }

return std::tuple<int, float>{A, B};
}

