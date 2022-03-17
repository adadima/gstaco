//
// Created by Alexandra Dima on 04.03.2022.
//
#include "einsum_taco/gstrt/tensor.h"
#include <gtest/gtest.h>

class TensorTest : public testing::Test {

};

TEST_F(TensorTest, Scalar) {
    auto s = Tensor<int, 0>({}, mode_dense);
    s.allocate();
    EXPECT_EQ(s.get(), 0);
    s.set(2);
    EXPECT_EQ(s.get(), 2);
}

TEST_F(TensorTest, DenseVector) {
    auto s = Tensor<int, 1>({5}, mode_dense);

    s.allocate();
    for (int i=0; i < 5; i++) {
        EXPECT_EQ(s.get({i}), 0);
    }

    for (int i=0; i < 5; i++) {
        s.set({i}, i + 1);
        EXPECT_EQ(s.get({i}), i + 1);
    }
}

TEST_F(TensorTest, DenseMatrix) {
    auto m = Tensor<float, 2>({10, 8}, mode_dense);
    m.allocate();

    for (int i=0; i < 10; i++) {
        for(int j=0; j < 8; j++) {
            m.set({i, j}, (i + j) / 1.0);
        }
    }

    for (int i=0; i < 10; i++) {
        for(int j=0; j < 8; j++) {
            EXPECT_EQ(m.get({i, j}), (i + j) / 1.0);
        }
    }
}

TEST_F(TensorTest, Dense3D) {

}

TEST_F(TensorTest, Dense4D) {

}

TEST_F(TensorTest, SparseMatrix) {
    auto s = Tensor<int, 2>({10, 20}, mode_sparse);
    s.allocate();
    EXPECT_EQ(s.compressed_size, 1);
    EXPECT_EQ(s.csr_data[0].rows.size(), 11);
    EXPECT_EQ(s.csr_data[0].cols.size(), 0);

    for (int i=0; i < 10; i++) {
        for (int j=0; j < 20; j++) {
            EXPECT_EQ(s.get({i, j}), 0);
        }
    }

    s.set({0, 1}, 1);
    ASSERT_EQ(s.csr_data[0].cols.size(), 1);
    EXPECT_EQ(s.csr_data[0].cols[0], 1);
    EXPECT_EQ(s.csr_data[0].rows[0], 0);
    EXPECT_EQ(s.csr_data[0].rows[1], 1);
    EXPECT_EQ(s.csr_data[0].vals[0], 1);
    EXPECT_EQ(s.get({0, 1}), 1);

    s.set({0, 1}, 2);
    EXPECT_EQ(s.csr_data[0].cols.size(), 1);
    EXPECT_EQ(s.csr_data[0].cols[0], 1);
    EXPECT_EQ(s.csr_data[0].rows[0], 0);
    EXPECT_EQ(s.csr_data[0].rows[1], 1);
    EXPECT_EQ(s.csr_data[0].vals[0], 2);
    EXPECT_EQ(s.get({0, 1}), 2);

    s.set({0, 4}, 1);
    EXPECT_EQ(s.csr_data[0].cols.size(), 2);
    EXPECT_EQ(s.csr_data[0].cols[1], 4);
    EXPECT_EQ(s.csr_data[0].rows[0], 0);
    EXPECT_EQ(s.csr_data[0].rows[1], 2);
    EXPECT_EQ(s.csr_data[0].vals[1], 1);
    EXPECT_EQ(s.get({0, 4}), 1);

    s.set({9, 19}, 1);
    EXPECT_EQ(s.csr_data[0].cols.size(), 3);
    EXPECT_EQ(s.csr_data[0].cols[2], 19);
    EXPECT_EQ(s.csr_data[0].rows[10], 3);
    for(int i=1; i < 10; i++) {
        EXPECT_EQ(s.csr_data[0].rows[i], 2);
    }
    EXPECT_EQ(s.csr_data[0].vals[2], 1);
    EXPECT_EQ(s.get({9, 19}), 1);
}

TEST_F(TensorTest, SparseMatrixLarge) {
    auto s = Tensor<int, 2>({1000000, 1000000}, mode_sparse);
    s.allocate();

    ASSERT_EQ(s.csr_data[0].rows.size(), 1000001);
    for (int i=0; i < 1000000; i += 1000) {
        s.set({5000, i}, 1);
        ASSERT_EQ(s.get({5000, i}), 1);
        ASSERT_EQ(s.get({4999, i}), 0);
    }
    std::cout << "Set round 1\n";

    ASSERT_EQ(s.csr_data[0].cols.size(), 1000);
    for (int i=0; i < 1000; i++) {
        ASSERT_EQ(s.csr_data[0].vals[i], 1);
    }
    s.set({5000, 0}, 2);
    ASSERT_EQ(s.get({5000, 0}), 2);
}