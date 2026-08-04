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

// Project header under test.
#include "fdtd/TMz/fdtd_tmz_grid2d.h"

// Standard library for limits, errors and sets.
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>

// Verify the accessors and the size of a grid.
TEST_CASE("Grid2D stores its geometry", "[unit][grid2d]")
{
    const fdtd::FloatType dx = 0.5;
    const fdtd::FloatType dy = 0.25;
    const fdtd::TMz::Grid2D grid(7, 5, dx, dy);
    CHECK(grid.nx() == 7);
    CHECK(grid.ny() == 5);
    CHECK(grid.dx() == dx);
    CHECK(grid.dy() == dy);
    CHECK(grid.pitch() == 7);
    CHECK(grid.size() == 35);
}

// Verify that indexing is row-major, x fastest, and covers every cell exactly once.
TEST_CASE("Grid2D indexing is row-major and one to one", "[unit][grid2d]")
{
    const auto [nx, ny] = GENERATE(
        table<std::size_t, std::size_t>({{1, 1}, {1, 9}, {9, 1}, {8, 8}, {7, 5}, {13, 31}}));
    const fdtd::TMz::Grid2D grid(nx, ny);
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
TEST_CASE("Grid2D rejects invalid geometry", "[unit][grid2d]")
{
    using fdtd::TMz::Grid2D;
    const auto huge = std::numeric_limits<std::size_t>::max();

    CHECK_THROWS_AS(Grid2D(0, 4), std::invalid_argument);
    CHECK_THROWS_AS(Grid2D(4, 0), std::invalid_argument);
    CHECK_THROWS_AS(Grid2D(4, 4, 0, 1), std::invalid_argument);
    CHECK_THROWS_AS(Grid2D(4, 4, 1, -1), std::invalid_argument);
    CHECK_THROWS_AS(Grid2D(4, 4, std::numeric_limits<fdtd::FloatType>::quiet_NaN(), 1),
                    std::invalid_argument);
    CHECK_THROWS_AS(Grid2D(huge, 2), std::invalid_argument);
}
