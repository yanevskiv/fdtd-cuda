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
#include <string>
#include <string_view>
#include <vector>

using fdtd::Argument;
using fdtd::split_arguments;
using fdtd::TMz::BackendType;
using fdtd::TMz::Boundary;
using fdtd::TMz::CudaKernel;
using fdtd::TMz::OmpRegion;
using fdtd::TMz::OmpSchedule;
using fdtd::TMz::Simulation;

namespace
{
    // Parse a variant name into default options.
    Simulation::Options variant(std::string_view text)
    {
        Simulation::Options options;
        fdtd::parse_omp_variant_arg("--variant", text, options);
        return options;
    }

    // Parse a CUDA variant name into default options.
    Simulation::Options cuda_variant(std::string_view text)
    {
        Simulation::Options options;
        fdtd::parse_cuda_variant_arg("--variant", text, options);
        return options;
    }

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

// Verify that names read into options and canonical names read back to the same options.
TEST_CASE("OpenMP variant names round-trip", "[unit][args]")
{
    const Simulation::Options rows = variant("default");
    CHECK(rows.omp_schedule == OmpSchedule::Rows);
    CHECK(fdtd::omp_variant_name(rows) == "rows+simd");

    const Simulation::Options tiles = variant("tiles:64x16+phase+simd");
    CHECK(tiles.omp_schedule == OmpSchedule::Tiles);
    CHECK(tiles.omp_tile_x == 64);
    CHECK(tiles.omp_tile_y == 16);
    CHECK(tiles.omp_simd);
    CHECK(tiles.omp_region == OmpRegion::Phase);
    CHECK(tiles.omp_threads == 0);
    CHECK(fdtd::omp_variant_name(tiles) == "tiles:64x16+simd+phase");

    for (const std::string name : {"rows", "rows+simd", "collapse2+phase", "tiles:128x32",
                                   "tasks:8x4+simd", "collapse2+simd+phase"}) {
        CAPTURE(name);
        CHECK(fdtd::omp_variant_name(variant(name)) == name);
    }
    CHECK(fdtd::omp_variant_name(variant("tasks")) == "tasks:128x32");
}

// Verify that a variant sets the omp_ fields but the thread count and the other settings.
TEST_CASE("OpenMP variant names leave the other options alone", "[unit][args]")
{
    Simulation::Options options;
    options.omp_threads = 3;
    options.boundary = Boundary::Pml;
    options.omp_tile_x = 7;
    fdtd::parse_omp_variant_arg("--variant", "rows", options);
    CHECK(options.omp_threads == 3);
    CHECK(options.boundary == Boundary::Pml);
    CHECK(options.omp_tile_x == 128);
    CHECK_FALSE(options.omp_simd);
}

// Verify that malformed names throw and leave the options unchanged.
TEST_CASE("OpenMP variant names reject malformed text", "[unit][args]")
{
    for (const char *name :
         {"", "fast", "Rows", "rows:", "rows:4x4", "collapse2:4x4", "tiles:", "tiles:4",
          "tiles:0x4", "tiles:4x0", "tiles:4x-1", "tiles:4x4x4", "tiles:ax4", "rows+", "rows+fast",
          "rows+simd+simd", "rows+phase+phase", "+simd", "default+simd"}) {
        CAPTURE(name);
        Simulation::Options options;
        options.omp_schedule = OmpSchedule::Tasks;
        CHECK_THROWS_AS(fdtd::parse_omp_variant_arg("--variant", name, options),
                        std::invalid_argument);
        CHECK(options.omp_schedule == OmpSchedule::Tasks);
    }
}

// Verify that CUDA names read into options and canonical names, block included, read back.
TEST_CASE("CUDA variant names round-trip", "[unit][args]")
{
    const Simulation::Options fused = cuda_variant("default");
    CHECK(fused.cuda_kernel == CudaKernel::Fused);
    CHECK(fused.cuda_block_x == 32);
    CHECK(fused.cuda_block_y == 8);
    CHECK(fdtd::cuda_variant_name(fused) == "fused:32x8");

    const Simulation::Options split = cuda_variant("split:16x4");
    CHECK(split.cuda_kernel == CudaKernel::Split);
    CHECK(split.cuda_block_x == 16);
    CHECK(split.cuda_block_y == 4);
    CHECK(fdtd::cuda_variant_name(split) == "split:16x4");

    CHECK(fdtd::cuda_variant_name(cuda_variant("split")) == "split:32x8");
    CHECK_FALSE(fused.cuda_graph);
    const Simulation::Options graph = cuda_variant("fused+graph");
    CHECK(graph.cuda_graph);
    CHECK(fdtd::cuda_variant_name(graph) == "fused:32x8+graph");
    for (const std::string name :
         {"fused:32x8", "fused:1x1", "split:128x1", "split:7x5", "split:16x16+graph"}) {
        CAPTURE(name);
        CHECK(fdtd::cuda_variant_name(cuda_variant(name)) == name);
    }
}

// Verify that a CUDA variant sets the cuda_ fields only, and that malformed names throw.
TEST_CASE("CUDA variant names reject malformed text", "[unit][args]")
{
    Simulation::Options options;
    options.omp_schedule = OmpSchedule::Tasks;
    options.cuda_block_x = 7;
    options.cuda_graph = true;
    fdtd::parse_cuda_variant_arg("--variant", "split", options);
    CHECK(options.omp_schedule == OmpSchedule::Tasks);
    CHECK(options.cuda_block_x == 32);
    CHECK_FALSE(options.cuda_graph);

    for (const char *name :
         {"", "Fused", "fast", "rows", "fused:", "fused:32", "fused:0x8", "fused:32x0",
          "fused:32x8x2", "fused:ax8", "fused+simd", "split:", "default:32x8", "fused+graph+graph",
          "fused+", "+graph", "fused+graph:32x8", "default+graph"}) {
        CAPTURE(name);
        Simulation::Options unchanged;
        unchanged.cuda_kernel = CudaKernel::Split;
        CHECK_THROWS_AS(fdtd::parse_cuda_variant_arg("--variant", name, unchanged),
                        std::invalid_argument);
        CHECK(unchanged.cuda_kernel == CudaKernel::Split);
    }
}

// Verify which backend reads which name, and the name each writes back.
TEST_CASE("Variant names belong to their backend", "[unit][args]")
{
    for (const BackendType backend :
         {BackendType::Serial, BackendType::OpenMP, BackendType::Cuda}) {
        CHECK(fdtd::backend_reads_variant(backend, "default"));
        CHECK_FALSE(fdtd::backend_reads_variant(backend, "fast"));
    }
    CHECK(fdtd::backend_reads_variant(BackendType::OpenMP, "tiles:8x8+simd"));
    CHECK_FALSE(fdtd::backend_reads_variant(BackendType::Cuda, "tiles:8x8+simd"));
    CHECK_FALSE(fdtd::backend_reads_variant(BackendType::Serial, "rows"));
    CHECK(fdtd::backend_reads_variant(BackendType::Cuda, "split:16x16"));
    CHECK_FALSE(fdtd::backend_reads_variant(BackendType::OpenMP, "split:16x16"));
    CHECK_FALSE(fdtd::backend_reads_variant(BackendType::Serial, "fused"));

    Simulation::Options options;
    fdtd::parse_variant_arg("--variant", BackendType::Cuda, "split:8x4", options);
    CHECK(options.cuda_kernel == CudaKernel::Split);
    fdtd::parse_variant_arg("--variant", BackendType::OpenMP, "collapse2", options);
    CHECK(options.omp_schedule == OmpSchedule::Collapse2);
    CHECK(options.cuda_kernel == CudaKernel::Split);
    CHECK_THROWS_AS(fdtd::parse_variant_arg("--variant", BackendType::Serial, "rows", options),
                    std::invalid_argument);
    CHECK_THROWS_AS(fdtd::parse_variant_arg("--variant", BackendType::OpenMP, "fused", options),
                    std::invalid_argument);

    CHECK(fdtd::variant_name(BackendType::Serial, options) == "default");
    CHECK(fdtd::variant_name(BackendType::OpenMP, options) == "collapse2");
    CHECK(fdtd::variant_name(BackendType::Cuda, options) == "split:8x4");
}
