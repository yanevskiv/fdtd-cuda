/*
 * C++ file for the tests of the CUDA backend's settings and variant names.
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

// Catch2 declarations for the tests.
#include <catch2/catch_test_macros.hpp>

// Project headers under test.
#include "fdtd/TMz/backends/fdtd_tmz_backend_cuda_options.h"

// Standard library for text and errors.
#include <stdexcept>
#include <string>

// Shorthand for the types under test.
using fdtd::TMz::backends::CudaLevel;
using fdtd::TMz::backends::CudaOptions;

// Verify that names read into settings and canonical names, block included, read back.
TEST_CASE("CUDA variant names round-trip", "[unit][cuda][options]")
{
    const CudaOptions defaults = CudaOptions::parse("default");
    CHECK(defaults.level == CudaLevel::V4);
    CHECK(defaults.block_x == 32);
    CHECK(defaults.block_y == 8);
    CHECK_FALSE(defaults.async_snapshots);
    CHECK(defaults.name() == "v4:32x8");

    const CudaOptions small = CudaOptions::parse("v2:16x4");
    CHECK(small.level == CudaLevel::V2);
    CHECK(small.block_x == 16);
    CHECK(small.block_y == 4);
    CHECK(small.name() == "v2:16x4");

    CHECK(CudaOptions::parse("v1").name() == "v1:32x8");
    CHECK(CudaOptions::parse("v3+async").async_snapshots);
    CHECK(CudaOptions::parse("v3+async").name() == "v3:32x8+async");
    CHECK_FALSE(CudaOptions::parse("v3").async_snapshots);
    for (const std::string name : {"v1:32x8", "v1:1x1", "v2:128x1", "v3:7x5", "v4:16x16",
                                   "v4:32x8+async", "v1:64x4+async"}) {
        CAPTURE(name);
        CHECK(CudaOptions::parse(name).name() == name);
    }
}

// Verify that malformed names and the names before the levels throw.
TEST_CASE("CUDA variant names reject malformed text", "[unit][cuda][options]")
{
    for (const char *name : {"",
                             "V1",
                             "fast",
                             "v0",
                             "v5",
                             "v1:",
                             "v1:32",
                             "v1:0x8",
                             "v1:32x0",
                             "v1:32x8x2",
                             "v1:ax8",
                             "v1+close",
                             "v1+async+async",
                             "v1+",
                             "+async",
                             "v1+async:32x8",
                             "default:32x8",
                             "default+async",
                             "fused",
                             "tiled:32x8",
                             "tiled+table+graph",
                             "v4+table",
                             "v2+graph"}) {
        CAPTURE(name);
        CHECK_THROWS_AS(CudaOptions::parse(name), std::invalid_argument);
    }
}
