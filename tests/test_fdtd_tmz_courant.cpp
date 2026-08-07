/*
 * C++ file for the 2D TMz Courant time step tests.
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
#include "fdtd/TMz/fdtd_tmz_courant.h"
#include "fdtd/fdtd_constants.h"

// Standard library for math, limits and errors.
#include <cmath>
#include <limits>
#include <stdexcept>

// Shorthand for the types under test.
using Catch::Matchers::WithinRel;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Grid2D;

// Verify the square-cell limit dx / (c0 * sqrt(2)) and its scaling with S.
TEST_CASE("courant_dt on square cells", "[unit][courant]")
{
    const Grid2D grid(8, 8, 1e-3F, 1e-3F);
    const double dx = double(grid.dx());
    const double limit = dx / (fdtd::C_0 * std::sqrt(2.0));

    CHECK_THAT(courant_dt(grid, 1.0), WithinRel(limit, 1e-12));
    CHECK_THAT(courant_dt(grid, 0.5), WithinRel(0.5 * limit, 1e-12));
}

// Verify rectangular cells against the general formula and that the limit lies
// below the 1D limit of either side.
TEST_CASE("courant_dt on rectangular cells", "[unit][courant]")
{
    const Grid2D grid(8, 8, 2e-3F, 1e-3F);
    const double dx = double(grid.dx());
    const double dy = double(grid.dy());
    const double limit = 1.0 / (fdtd::C_0 * std::sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy)));
    const double dt = courant_dt(grid, 0.99);

    CHECK_THAT(dt, WithinRel(0.99 * limit, 1e-12));
    CHECK(dt < dy / fdtd::C_0);
}

// Verify that Courant numbers outside (0, 1] are rejected.
TEST_CASE("courant_dt rejects invalid Courant numbers", "[unit][courant]")
{
    const Grid2D grid(4, 4);

    CHECK_THROWS_AS(courant_dt(grid, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(courant_dt(grid, -0.5), std::invalid_argument);
    CHECK_THROWS_AS(courant_dt(grid, 1.01), std::invalid_argument);
    CHECK_THROWS_AS(courant_dt(grid, std::numeric_limits<double>::quiet_NaN()),
                    std::invalid_argument);
    CHECK_NOTHROW(courant_dt(grid, 1.0));
}
