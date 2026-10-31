// tests/unit/test_numbers.cpp
// Tests for the non-throwing Nastran number parsers in core/numbers.hpp.
// The parsers must accept every legal Nastran form and reject anything that
// is not a whole-field number, so callers never silently use a prefix.

#include <gtest/gtest.h>
#include "core/numbers.hpp"

using namespace vibestran;

TEST(Numbers, IntegerAcceptsSignedWholeFields) {
    EXPECT_EQ(parse_integer("42"), 42);
    EXPECT_EQ(parse_integer(" 7 "), 7);
    EXPECT_EQ(parse_integer("+5"), 5);
    EXPECT_EQ(parse_integer("-123"), -123);
    EXPECT_EQ(parse_integer("2147483647"), 2147483647);
}

TEST(Numbers, IntegerRejectsPartialAndNonIntegerFields) {
    // std::stoi would have returned 12, 1 and 3 for these.
    EXPECT_FALSE(parse_integer("12.5"));
    EXPECT_FALSE(parse_integer("1abc"));
    EXPECT_FALSE(parse_integer("3 4"));
    EXPECT_FALSE(parse_integer(""));
    EXPECT_FALSE(parse_integer("   "));
    EXPECT_FALSE(parse_integer("+"));
    EXPECT_FALSE(parse_integer("++1"));
    EXPECT_FALSE(parse_integer("2147483648"));
}

TEST(Numbers, RealAcceptsAllNastranForms) {
    EXPECT_DOUBLE_EQ(*parse_real("1.5E+3"), 1500.0);
    EXPECT_DOUBLE_EQ(*parse_real("1.5e3"), 1500.0);
    EXPECT_DOUBLE_EQ(*parse_real("1.5D-3"), 1.5e-3);
    EXPECT_DOUBLE_EQ(*parse_real("1.5d+3"), 1500.0);
    // Implicit exponent: the sign after the mantissa starts the exponent.
    EXPECT_DOUBLE_EQ(*parse_real("1.5+3"), 1500.0);
    EXPECT_DOUBLE_EQ(*parse_real("1.5-3"), 1.5e-3);
    EXPECT_DOUBLE_EQ(*parse_real("-2.-1"), -0.2);
    EXPECT_DOUBLE_EQ(*parse_real("1.+7"), 1.0e7);
    EXPECT_DOUBLE_EQ(*parse_real(".5"), 0.5);
    EXPECT_DOUBLE_EQ(*parse_real("5."), 5.0);
    EXPECT_DOUBLE_EQ(*parse_real("+12"), 12.0);
    EXPECT_DOUBLE_EQ(*parse_real(" -0.25 "), -0.25);
}

TEST(Numbers, RealRejectsMalformedAndNonFiniteFields) {
    // std::stod would have returned 1.5 and 1.0 for the first two.
    EXPECT_FALSE(parse_real("1.5abc"));
    EXPECT_FALSE(parse_real("1.5E3E2"));
    EXPECT_FALSE(parse_real(""));
    EXPECT_FALSE(parse_real("+"));
    EXPECT_FALSE(parse_real("E5"));
    EXPECT_FALSE(parse_real("abc"));
    EXPECT_FALSE(parse_real("inf"));
    EXPECT_FALSE(parse_real("nan"));
    EXPECT_FALSE(parse_real("1e400"));
    EXPECT_FALSE(parse_real("1.5+"));
}
