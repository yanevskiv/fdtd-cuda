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
#include "fdtd/TMz/fdtd_tmz_sim2d.h"

// Shorthand for the types under test.
using fdtd::TMz::OmpOptions;
using fdtd::TMz::OmpRegion;
using fdtd::TMz::OmpSchedule;

// Verify the defaults: rows, the OpenMP thread count, 128 by 32 tiles, no SIMD, one region.
TEST_CASE("OpenMP options defaults", "[unit][omp_options]")
{
    const OmpOptions opts;

    CHECK(opts.schedule == OmpSchedule::Rows);
    CHECK(opts.threads == 0);
    CHECK(opts.tile_x == 128);
    CHECK(opts.tile_y == 32);
    CHECK_FALSE(opts.simd);
    CHECK(opts.region == OmpRegion::Run);
}

// Verify that a simulation carries the OpenMP options it was given.
TEST_CASE("Sim2D keeps the OpenMP options", "[unit][omp_options]")
{
    fdtd::TMz::Sim2D::Options opts;
    opts.omp.schedule = OmpSchedule::Tiles;
    opts.omp.tile_x = 16;
    const fdtd::TMz::Sim2D sim(8, 8, opts);

    CHECK(sim.options().omp.schedule == OmpSchedule::Tiles);
    CHECK(sim.options().omp.tile_x == 16);
}
