/*
 * C++ file for tests of the physical constants.
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
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers under test.
#include "fdtd/fdtd.h"

// Standard library for the square root.
#include <cmath>

using Catch::Matchers::WithinRel;

// Verify the constants against their reference values.
TEST_CASE("Physical constants have their SI values", "[unit][constants]")
{
    CHECK(fdtd::C_0 == 299792458.0);
    CHECK_THAT(fdtd::MU_0, WithinRel(1.25663706127e-6, 1e-12));
    CHECK_THAT(fdtd::EPS_0, WithinRel(8.8541878188e-12, 1e-10));
    CHECK_THAT(fdtd::ETA_0, WithinRel(376.730313412, 1e-10));
}

// Verify that the constants agree with each other.
TEST_CASE("Physical constants are consistent", "[unit][constants]")
{
    CHECK_THAT(1.0 / std::sqrt(fdtd::MU_0 * fdtd::EPS_0), WithinRel(fdtd::C_0, 1e-14));
    CHECK_THAT(std::sqrt(fdtd::MU_0 / fdtd::EPS_0), WithinRel(fdtd::ETA_0, 1e-14));
}
