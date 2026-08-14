/*
 * C++ file for the command-line argument parsing tests.
 *
 * Copyright (C) 2026 Ivan Janevski
 *
 * fdtd-cuda  is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free  Software  Foundation;  either  version 3 of the License, or (at
 * your option) any later version.
 *
 * fdtd-cuda  is  distributed  in  the  hope that it will be useful, but
 * WITHOUT   ANY   WARRANTY;   without  even  the  implied  warranty  of
 * MERCHANTABILITY  or  FITNESS  FOR  A  PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You  should  have  received  a copy of the GNU General Public License
 * along with fdtd-cuda; if not, see <https://www.gnu.org/licenses/>.
 */

#include <catch2/catch_test_macros.hpp>

#include "fdtd/fdtd_args.h"

#include <stdexcept>
#include <string_view>
#include <vector>

using fdtd::Argument;
using fdtd::split_arguments;
using fdtd::TMz::Boundary;
using fdtd::TMz::backends::BackendType;

namespace
{
    // Split a list of arguments with -h and --quiet as the flags.
    std::vector<Argument> split(std::vector<std::string_view> args)
    {
        static constexpr std::string_view FLAGS[] = {"-h", "--quiet"};
        return split_arguments(args, FLAGS);
    }
}

// Verify that options take a value after '=' or as the next argument, and flags none.
TEST_CASE("Arguments split into options and flags", "[unit][args]")
{
    const auto options = split({"--nx", "64", "-h", "--ny=32", "--quiet", "--name=", "--x=a=b"});
    REQUIRE(options.size() == 6);
    CHECK(options[0].name == "--nx");
    CHECK(options[0].value == "64");
    CHECK(options[1].name == "-h");
    CHECK(options[1].value.empty());
    CHECK(options[2].name == "--ny");
    CHECK(options[2].value == "32");
    CHECK(options[3].name == "--quiet");
    CHECK(options[4].name == "--name");
    CHECK(options[4].value.empty());
    CHECK(options[5].name == "--x");
    CHECK(options[5].value == "a=b");
    CHECK(split({}).empty());
}

// Verify that stray arguments, missing values and flags with values throw.
TEST_CASE("Arguments reject bad input", "[unit][args]")
{
    CHECK_THROWS_AS(split({"stray"}), std::invalid_argument);
    CHECK_THROWS_AS(split({"-x"}), std::invalid_argument);
    CHECK_THROWS_AS(split({"--nx"}), std::invalid_argument);
    CHECK_THROWS_AS(split({"--quiet=yes"}), std::invalid_argument);
}

// Verify the number parsers on good and bad values.
TEST_CASE("Argument numbers parse whole values only", "[unit][args]")
{
    CHECK(fdtd::parse_size_arg("--n", "0") == 0);
    CHECK(fdtd::parse_size_arg("--n", "4096") == 4096);
    CHECK_THROWS_AS(fdtd::parse_size_arg("--n", ""), std::invalid_argument);
    CHECK_THROWS_AS(fdtd::parse_size_arg("--n", "-1"), std::invalid_argument);
    CHECK_THROWS_AS(fdtd::parse_size_arg("--n", "12x"), std::invalid_argument);

    CHECK(fdtd::parse_positive_arg("--x", "2.5") == 2.5);
    CHECK(fdtd::parse_positive_arg("--x", "1e-3") == 1e-3);
    CHECK_THROWS_AS(fdtd::parse_positive_arg("--x", "0"), std::invalid_argument);
    CHECK_THROWS_AS(fdtd::parse_positive_arg("--x", "-1"), std::invalid_argument);
    CHECK_THROWS_AS(fdtd::parse_positive_arg("--x", "inf"), std::invalid_argument);
    CHECK_THROWS_AS(fdtd::parse_positive_arg("--x", "nan"), std::invalid_argument);

    CHECK(fdtd::parse_nonnegative_arg("--x", "0") == 0);
    CHECK(fdtd::parse_nonnegative_arg("--x", "0.1") == 0.1);
    CHECK_THROWS_AS(fdtd::parse_nonnegative_arg("--x", "-0.1"), std::invalid_argument);
    CHECK_THROWS_AS(fdtd::parse_nonnegative_arg("--x", "inf"), std::invalid_argument);
    CHECK_THROWS_AS(fdtd::parse_nonnegative_arg("--x", "1s"), std::invalid_argument);
}

// Verify that a list splits at every comma and keeps empty fields.
TEST_CASE("Argument lists split at commas", "[unit][args]")
{
    CHECK(fdtd::split_list_arg("1,2,3") == std::vector<std::string_view>{"1", "2", "3"});
    CHECK(fdtd::split_list_arg("a") == std::vector<std::string_view>{"a"});
    CHECK(fdtd::split_list_arg("a,,b,") == std::vector<std::string_view>{"a", "", "b", ""});
}

// Verify the boundary and backend names, lowercase only.
TEST_CASE("Argument names parse boundaries and backends", "[unit][args]")
{
    CHECK(fdtd::parse_boundary_arg("--b", "pec") == Boundary::Pec);
    CHECK(fdtd::parse_boundary_arg("--b", "pmc") == Boundary::Pmc);
    CHECK(fdtd::parse_boundary_arg("--b", "pml") == Boundary::Pml);
    CHECK_THROWS_AS(fdtd::parse_boundary_arg("--b", "PML"), std::invalid_argument);

    CHECK(fdtd::parse_backend_arg("--b", "serial") == BackendType::Serial);
    CHECK(fdtd::parse_backend_arg("--b", "openmp") == BackendType::OpenMP);
    CHECK(fdtd::parse_backend_arg("--b", "cuda") == BackendType::Cuda);
    CHECK_THROWS_AS(fdtd::parse_backend_arg("--b", "gpu"), std::invalid_argument);
}
