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
using fdtd::TMz::backends::OmpLevel;
using fdtd::TMz::backends::OmpOptions;

// Verify the defaults: the highest level, runtime threads and runtime binding.
TEST_CASE("OpenMP settings defaults", "[unit][openmp][options]")
{
    const OmpOptions options;

    CHECK(options.level == OmpLevel::V4);
    CHECK(options.threads == 0);
    CHECK(options.bind == OmpBind::Default);
}

// Verify that names read into settings and canonical names read back to the same settings.
TEST_CASE("OpenMP variant names round-trip", "[unit][openmp][options]")
{
    const OmpOptions defaults = OmpOptions::parse("default");
    CHECK(defaults.level == OmpLevel::V4);
    CHECK(defaults.name() == "v4");

    const OmpOptions plain = OmpOptions::parse("v1");
    CHECK(plain.level == OmpLevel::V1);
    CHECK(plain.bind == OmpBind::Default);
    CHECK(plain.threads == 0);

    const OmpOptions spread = OmpOptions::parse("v3+spread");
    CHECK(spread.level == OmpLevel::V3);
    CHECK(spread.bind == OmpBind::Spread);

    for (const std::string name : {"v1", "v2", "v3", "v4", "v1+close", "v2+spread", "v4+close"}) {
        CAPTURE(name);
        CHECK(OmpOptions::parse(name).name() == name);
    }
}

// Verify that malformed names and the names before the levels throw.
TEST_CASE("OpenMP variant names reject malformed text", "[unit][openmp][options]")
{
    for (const char *name : {"",
                             "fast",
                             "V1",
                             "v0",
                             "v5",
                             "v10",
                             "v",
                             "v1+",
                             "v1+fast",
                             "v1+close+spread",
                             "v1+close+close",
                             "+close",
                             "default+close",
                             "rows",
                             "rows+simd",
                             "rows+simd+table+fuse",
                             "collapse2",
                             "v4+fuse",
                             "v2+phase",
                             "v2:4x4"}) {
        CAPTURE(name);
        CHECK_THROWS_AS(OmpOptions::parse(name), std::invalid_argument);
    }
}
