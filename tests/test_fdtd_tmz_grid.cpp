/*
 * C++ file for the 2D TMz grid tests.
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
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers under test.
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/fdtd_constants.h"

// Standard library for math, limits, errors and sets.
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>

// Shorthand for the types under test.
using Catch::Matchers::WithinRel;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Grid;

// Verify the accessors and the size of a grid.
TEST_CASE("Grid stores its geometry", "[unit][grid]")
{
    const fdtd::FloatType dx = 0.5;
    const fdtd::FloatType dy = 0.25;
    const fdtd::TMz::Grid grid(7, 5, dx, dy);
    CHECK(grid.nx() == 7);
    CHECK(grid.ny() == 5);
    CHECK(grid.dx() == dx);
    CHECK(grid.dy() == dy);
    CHECK(grid.pitch() == 7);
    CHECK(grid.size() == 35);
}

// Verify that indexing is row-major, x fastest, and covers every cell exactly once.
TEST_CASE("Grid indexing is row-major and one to one", "[unit][grid]")
{
    const auto [nx, ny] = GENERATE(table<std::size_t, std::size_t>({{1, 1}, {1, 9}, {9, 1}, {8, 8}, {7, 5}, {13, 31}}));
    const fdtd::TMz::Grid grid(nx, ny);
    std::set<std::size_t> seen;

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            CHECK(grid.index(i, j) == j * grid.pitch() + i);
            seen.insert(grid.index(i, j));
        }
    }
    CHECK(seen.size() == nx * ny);
    CHECK(grid.index(0, 0) == 0);
    CHECK(grid.index(nx - 1, 0) == nx - 1);
    CHECK(grid.index(0, ny - 1) == (ny - 1) * grid.pitch());
    CHECK(grid.index(nx - 1, ny - 1) == grid.size() - 1);
}

// Verify that invalid geometries are rejected.
TEST_CASE("Grid rejects invalid geometry", "[unit][grid]")
{
    using fdtd::TMz::Grid;
    const auto huge = std::numeric_limits<std::size_t>::max();

    CHECK_THROWS_AS(Grid(0, 4), std::invalid_argument);
    CHECK_THROWS_AS(Grid(4, 0), std::invalid_argument);
    CHECK_THROWS_AS(Grid(4, 4, 0, 1), std::invalid_argument);
    CHECK_THROWS_AS(Grid(4, 4, 1, -1), std::invalid_argument);
    CHECK_THROWS_AS(Grid(4, 4, std::numeric_limits<fdtd::FloatType>::quiet_NaN(), 1), std::invalid_argument);
    CHECK_THROWS_AS(Grid(huge, 2), std::invalid_argument);
}

// Verify the square-cell limit dx / (c0 * sqrt(2)) and its scaling with S.
TEST_CASE("courant_dt on square cells", "[unit][courant]")
{
    const Grid grid(8, 8, 1e-3F, 1e-3F);
    const double dx = double(grid.dx());
    const double limit = dx / (fdtd::C_0 * std::sqrt(2.0));

    CHECK_THAT(courant_dt(grid, 1.0), WithinRel(limit, 1e-12));
    CHECK_THAT(courant_dt(grid, 0.5), WithinRel(0.5 * limit, 1e-12));
}

// Verify rectangular cells against the formula and below the 1D limit of either side.
TEST_CASE("courant_dt on rectangular cells", "[unit][courant]")
{
    const Grid grid(8, 8, 2e-3F, 1e-3F);
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
    const Grid grid(4, 4);

    CHECK_THROWS_AS(courant_dt(grid, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(courant_dt(grid, -0.5), std::invalid_argument);
    CHECK_THROWS_AS(courant_dt(grid, 1.01), std::invalid_argument);
    CHECK_THROWS_AS(courant_dt(grid, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    CHECK_NOTHROW(courant_dt(grid, 1.0));
}
