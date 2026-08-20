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
using fdtd::TMz::backends::CudaCoeffs;
using fdtd::TMz::backends::CudaKernel;
using fdtd::TMz::backends::CudaOptions;

// Verify that names read into settings and canonical names, block included, read back.
TEST_CASE("CUDA variant names round-trip", "[unit][cuda][options]")
{
    const CudaOptions defaults = CudaOptions::parse("default");
    CHECK(defaults.kernel == CudaKernel::Tiled);
    CHECK(defaults.block_x == 32);
    CHECK(defaults.block_y == 8);
    CHECK(defaults.graph);
    CHECK(defaults.coeffs == CudaCoeffs::Table);
    CHECK(defaults.name() == "tiled:32x8+table+graph");

    const CudaOptions tiled = CudaOptions::parse("tiled:16x4");
    CHECK(tiled.kernel == CudaKernel::Tiled);
    CHECK(tiled.block_x == 16);
    CHECK(tiled.block_y == 4);
    CHECK(tiled.coeffs == CudaCoeffs::Arrays);
    CHECK_FALSE(tiled.graph);
    CHECK(tiled.name() == "tiled:16x4");

    CHECK(CudaOptions::parse("tiled").name() == "tiled:32x8");
    const CudaOptions graph = CudaOptions::parse("fused+graph");
    CHECK(graph.kernel == CudaKernel::Fused);
    CHECK(graph.graph);
    CHECK(graph.coeffs == CudaCoeffs::Arrays);
    CHECK(graph.name() == "fused:32x8+graph");
    const CudaOptions both = CudaOptions::parse("tiled+graph+table");
    CHECK(both.coeffs == CudaCoeffs::Table);
    CHECK(both.graph);
    CHECK(both.name() == "tiled:32x8+table+graph");
    CHECK(CudaOptions::parse("fused+async").async_snapshots);
    CHECK_FALSE(CudaOptions::parse("fused+graph").async_snapshots);
    for (const std::string name :
         {"fused:32x8", "fused:1x1", "tiled:128x1", "tiled:7x5", "tiled:16x16+graph",
          "fused:32x8+table", "fused:64x4+table+graph", "tiled:32x8", "tiled:16x16+table+graph",
          "fused:32x8+async", "tiled:32x8+table+graph+async"}) {
        CAPTURE(name);
        CHECK(CudaOptions::parse(name).name() == name);
    }
}

// Verify that malformed names throw.
TEST_CASE("CUDA variant names reject malformed text", "[unit][cuda][options]")
{
    for (const char *name : {"",
                             "Fused",
                             "fast",
                             "rows",
                             "fused:",
                             "fused:32",
                             "fused:0x8",
                             "fused:32x0",
                             "fused:32x8x2",
                             "fused:ax8",
                             "fused+simd",
                             "split",
                             "split:32x8",
                             "default:32x8",
                             "fused+graph+graph",
                             "fused+",
                             "+graph",
                             "fused+graph:32x8",
                             "default+graph",
                             "fused+table+table",
                             "fused+tables",
                             "fused+async+async"}) {
        CAPTURE(name);
        CHECK_THROWS_AS(CudaOptions::parse(name), std::invalid_argument);
    }
}
