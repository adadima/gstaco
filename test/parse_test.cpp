//
// Created by Alexandra Dima on 19.12.2021.
//
#include <iostream>
#include <gtest/gtest.h>
#include "einsum_taco/ir/ir.h"
#include <einsum_taco/parser/heading.h>


class ParseTest : public testing::Test {
public:
    einsum::Module parse(std::string code) {
        char temp_name[17];
        strcpy(temp_name, "dump_test_XXXXXX");
        int f = mkstemp(temp_name);
        if (f == -1) {
            throw std::system_error(errno, std::system_category());
        }
        auto temp = fdopen(f, "r+");
        auto num_written = fwrite(code.data(), 1, code.size(), temp);
        if (num_written != code.size()) {
            throw std::runtime_error("Did not write correct bytes");
        }
        auto status = fseek(temp, 0, SEEK_SET);
        auto module = parse_module(temp);

        fclose(temp);
        close(f);
        return module;
    }
};

TEST_F(ParseTest, LiteralsTest) {
    EXPECT_EQ (parse("6").dump(),  "6\n");

    EXPECT_EQ (parse("true").dump(),  "true\n");

    EXPECT_EQ (parse("6.0").dump().rfind("6.0", 0),  0);
}

