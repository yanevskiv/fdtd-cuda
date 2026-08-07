/*
 * C++ file for the 2D TMz material map tests.
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
#include "fdtd/TMz/fdtd_tmz_material2d.h"

// Standard library for limits and errors.
#include <cstddef>
#include <limits>
#include <stdexcept>

// Shorthand for the types under test.
using fdtd::TMz::Grid2D;
using fdtd::TMz::Material2D;

// Medium used to mark filled cells.
static const Material2D::Medium GLASS{4, 1, 0.5, 0};

// Verify that a new material is vacuum everywhere.
TEST_CASE("Material2D starts as vacuum", "[unit][material2d]")
{
    const Material2D mat(Grid2D(7, 5));
    REQUIRE(mat.eps_r().size() == mat.grid().size());

    for (std::size_t k = 0; k < mat.grid().size(); ++k) {
        CHECK(mat.eps_r()[k] == fdtd::FloatType(1));
        CHECK(mat.mu_r()[k] == fdtd::FloatType(1));
        CHECK(mat.sigma_e()[k] == fdtd::FloatType(0));
        CHECK(mat.sigma_m()[k] == fdtd::FloatType(0));
    }
}

// Verify that fill_rect sets exactly the half-open rectangle.
TEST_CASE("Material2D fill_rect sets the half-open rectangle", "[unit][material2d]")
{
    Material2D mat(Grid2D(8, 6));
    mat.fill_rect(2, 1, 5, 4, GLASS);

    for (std::size_t j = 0; j < 6; ++j) {
        for (std::size_t i = 0; i < 8; ++i) {
            const bool inside = i >= 2 && i < 5 && j >= 1 && j < 4;
            const auto m = mat.at(i, j);
            CHECK(m.eps_r == (inside ? GLASS.eps_r : fdtd::FloatType(1)));
            CHECK(m.sigma_e == (inside ? GLASS.sigma_e : fdtd::FloatType(0)));
        }
    }
}

// Verify that rectangles are clipped to the grid and empty ones change nothing.
TEST_CASE("Material2D fill_rect clips to the grid", "[unit][material2d]")
{
    Material2D mat(Grid2D(4, 4));
    mat.fill_rect(3, 3, 100, 100, GLASS);
    mat.fill_rect(2, 0, 1, 4, GLASS);
    mat.fill_rect(10, 10, 20, 20, GLASS);

    std::size_t filled = 0;
    for (std::size_t k = 0; k < mat.grid().size(); ++k) {
        filled += mat.eps_r()[k] == GLASS.eps_r;
    }
    CHECK(filled == 1);
    CHECK(mat.at(3, 3).eps_r == GLASS.eps_r);
}

// Verify that fill sets every cell.
TEST_CASE("Material2D fill sets every cell", "[unit][material2d]")
{
    Material2D mat(Grid2D(3, 2));
    mat.fill(GLASS);

    for (std::size_t k = 0; k < mat.grid().size(); ++k) {
        CHECK(mat.eps_r()[k] == GLASS.eps_r);
        CHECK(mat.sigma_e()[k] == GLASS.sigma_e);
    }
}

// Verify fill_circle against the cell-center test, including a center off the grid.
TEST_CASE("Material2D fill_circle sets the cells within the radius", "[unit][material2d]")
{
    const double ci = GENERATE(4.0, 4.5, -1.0, 8.5);
    const double cj = 3.0;
    const double r = 2.5;
    Material2D mat(Grid2D(9, 7));
    mat.fill_circle(ci, cj, r, GLASS);

    for (std::size_t j = 0; j < 7; ++j) {
        for (std::size_t i = 0; i < 9; ++i) {
            const double x = double(i) - ci;
            const double y = double(j) - cj;
            const bool inside = x * x + y * y <= r * r;
            CHECK((mat.at(i, j).eps_r == GLASS.eps_r) == inside);
        }
    }
}

// Verify that a zero radius sets only the cell at the center.
TEST_CASE("Material2D fill_circle with zero radius sets one cell", "[unit][material2d]")
{
    Material2D mat(Grid2D(5, 5));
    mat.fill_circle(2, 3, 0, GLASS);

    std::size_t filled = 0;
    for (std::size_t k = 0; k < mat.grid().size(); ++k) {
        filled += mat.eps_r()[k] == GLASS.eps_r;
    }
    CHECK(filled == 1);
    CHECK(mat.at(2, 3).eps_r == GLASS.eps_r);
}

// Verify that unphysical media, bad circles and out-of-range cells are rejected.
TEST_CASE("Material2D rejects invalid input", "[unit][material2d]")
{
    Material2D mat(Grid2D(4, 4));
    const auto nan = std::numeric_limits<fdtd::FloatType>::quiet_NaN();
    const auto inf = std::numeric_limits<double>::infinity();

    CHECK_THROWS_AS(mat.fill({0, 1, 0, 0}), std::invalid_argument);
    CHECK_THROWS_AS(mat.fill({1, -1, 0, 0}), std::invalid_argument);
    CHECK_THROWS_AS(mat.fill({1, 1, -1, 0}), std::invalid_argument);
    CHECK_THROWS_AS(mat.fill({1, 1, 0, -1}), std::invalid_argument);
    CHECK_THROWS_AS(mat.fill({nan, 1, 0, 0}), std::invalid_argument);
    CHECK_THROWS_AS(mat.fill_rect(0, 0, 4, 4, {1, 1, nan, 0}), std::invalid_argument);
    CHECK_THROWS_AS(mat.fill_circle(1, 1, -1, GLASS), std::invalid_argument);
    CHECK_THROWS_AS(mat.fill_circle(inf, 1, 1, GLASS), std::invalid_argument);
    CHECK_THROWS_AS(mat.at(4, 0), std::out_of_range);
    CHECK_THROWS_AS(mat.at(0, 4), std::out_of_range);
    CHECK(mat.at(0, 0).eps_r == fdtd::FloatType(1));
}
