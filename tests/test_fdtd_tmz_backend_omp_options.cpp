/*
 * C++ file for the tests of the OpenMP backend's settings and variant names.
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
#include "fdtd/TMz/backends/fdtd_tmz_backend_omp_options.h"

// Standard library for text and errors.
#include <stdexcept>
#include <string>

// Shorthand for the types under test.
using fdtd::TMz::backends::OmpBind;
using fdtd::TMz::backends::OmpOptions;
using fdtd::TMz::backends::OmpRegion;
using fdtd::TMz::backends::OmpSchedule;

// Verify the defaults: rows, runtime threads, SIMD, table, fusion, one region, runtime binding.
TEST_CASE("OpenMP settings defaults", "[unit][openmp][options]")
{
    const OmpOptions options;

    CHECK(options.schedule == OmpSchedule::Rows);
    CHECK(options.threads == 0);
    CHECK(options.table);
    CHECK(options.fuse);
    CHECK(options.simd);
    CHECK(options.region == OmpRegion::Run);
    CHECK(options.bind == OmpBind::Default);
}

// Verify that names read into settings and canonical names read back to the same settings.
TEST_CASE("OpenMP variant names round-trip", "[unit][openmp][options]")
{
    const OmpOptions rows = OmpOptions::parse("default");
    CHECK(rows.schedule == OmpSchedule::Rows);
    CHECK(rows.name() == "rows+simd+table+fuse");

    const OmpOptions phase = OmpOptions::parse("collapse2+phase+simd");
    CHECK(phase.schedule == OmpSchedule::Collapse2);
    CHECK(phase.simd);
    CHECK(phase.region == OmpRegion::Phase);
    CHECK(phase.threads == 0);
    CHECK(phase.name() == "collapse2+simd+phase");

    const OmpOptions fused = OmpOptions::parse("rows+fuse+table");
    CHECK(fused.table);
    CHECK(fused.fuse);
    CHECK_FALSE(fused.simd);
    CHECK(fused.name() == "rows+table+fuse");

    for (const std::string name :
         {"rows", "rows+simd", "collapse2+phase", "collapse2", "collapse2+simd",
          "collapse2+simd+phase", "rows+table", "rows+simd+table+fuse", "collapse2+table+phase",
          "rows+fuse+phase", "rows+simd+table+fuse+close", "collapse2+spread"}) {
        CAPTURE(name);
        CHECK(OmpOptions::parse(name).name() == name);
    }
}

// Verify that malformed names throw.
TEST_CASE("OpenMP variant names reject malformed text", "[unit][openmp][options]")
{
    for (const char *name : {"",
                             "fast",
                             "Rows",
                             "tasks",
                             "tasks:8x8",
                             "tasks+phase",
                             "rows:",
                             "rows:4x4",
                             "collapse2:4x4",
                             "tiles:",
                             "tiles:4",
                             "tiles:0x4",
                             "tiles:4x0",
                             "tiles:4x-1",
                             "tiles:4x4x4",
                             "tiles:ax4",
                             "rows+",
                             "rows+fast",
                             "rows+simd+simd",
                             "rows+phase+phase",
                             "+simd",
                             "default+simd",
                             "rows+table+table",
                             "rows+simd+fuse+fuse",
                             "+table",
                             "rows+close+spread",
                             "rows+close+close",
                             "rows+spread+spread"}) {
        CAPTURE(name);
        CHECK_THROWS_AS(OmpOptions::parse(name), std::invalid_argument);
    }
}
