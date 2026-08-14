/*
 * C++ file for the tests of the 2D TMz OpenMP backend options.
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
#include "fdtd/TMz/fdtd_tmz_omp_options.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"

// Standard library for errors and strings.
#include <stdexcept>
#include <string>

// Shorthand for the types under test.
using fdtd::TMz::OmpOptions;
using fdtd::TMz::OmpRegion;
using fdtd::TMz::OmpSchedule;

// Verify the defaults: rows, the OpenMP thread count, 128 by 32 tiles, SIMD, one region.
TEST_CASE("OpenMP options defaults", "[unit][omp_options]")
{
    const OmpOptions opts;

    CHECK(opts.schedule == OmpSchedule::Rows);
    CHECK(opts.threads == 0);
    CHECK(opts.tile_x == 128);
    CHECK(opts.tile_y == 32);
    CHECK(opts.simd);
    CHECK(opts.region == OmpRegion::Run);
}

// Verify that a simulation carries the OpenMP options it was given.
TEST_CASE("Simulation keeps the OpenMP options", "[unit][omp_options]")
{
    fdtd::TMz::Simulation::Options opts;
    opts.omp.schedule = OmpSchedule::Tiles;
    opts.omp.tile_x = 16;
    const fdtd::TMz::Simulation sim(8, 8, opts);

    CHECK(sim.options().omp.schedule == OmpSchedule::Tiles);
    CHECK(sim.options().omp.tile_x == 16);
}

// Verify that names read into options and canonical names read back to the same options.
TEST_CASE("OpenMP variant names round-trip", "[unit][omp_options]")
{
    const OmpOptions rows = fdtd::TMz::parse_omp_variant("default");
    CHECK(rows.schedule == OmpSchedule::Rows);
    CHECK(fdtd::TMz::omp_variant_name(rows) == "rows+simd");

    const OmpOptions tiles = fdtd::TMz::parse_omp_variant("tiles:64x16+phase+simd");
    CHECK(tiles.schedule == OmpSchedule::Tiles);
    CHECK(tiles.tile_x == 64);
    CHECK(tiles.tile_y == 16);
    CHECK(tiles.simd);
    CHECK(tiles.region == OmpRegion::Phase);
    CHECK(tiles.threads == 0);
    CHECK(fdtd::TMz::omp_variant_name(tiles) == "tiles:64x16+simd+phase");

    for (const std::string name : {"rows", "rows+simd", "collapse2+phase", "tiles:128x32",
                                   "tasks:8x4+simd", "collapse2+simd+phase"}) {
        CAPTURE(name);
        CHECK(fdtd::TMz::omp_variant_name(fdtd::TMz::parse_omp_variant(name)) == name);
    }
    CHECK(fdtd::TMz::omp_variant_name(fdtd::TMz::parse_omp_variant("tasks")) == "tasks:128x32");
}

// Verify that malformed names throw.
TEST_CASE("OpenMP variant names reject malformed text", "[unit][omp_options]")
{
    for (const char *name :
         {"", "fast", "Rows", "rows:", "rows:4x4", "collapse2:4x4", "tiles:", "tiles:4",
          "tiles:0x4", "tiles:4x0", "tiles:4x-1", "tiles:4x4x4", "tiles:ax4", "rows+", "rows+fast",
          "rows+simd+simd", "rows+phase+phase", "+simd", "default+simd"}) {
        CAPTURE(name);
        CHECK_THROWS_AS(fdtd::TMz::parse_omp_variant(name), std::invalid_argument);
    }
}
