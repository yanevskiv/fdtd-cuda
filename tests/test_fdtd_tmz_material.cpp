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
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers under test.
#include "fdtd/TMz/fdtd_tmz_material.h"
#include "fdtd/fdtd_constants.h"

// Standard library for math, limits and errors.
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

// Shorthand for the types under test.
using Catch::Matchers::WithinRel;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;

// Medium used to mark filled cells.
static const Material::Medium GLASS{4, 1, 0.5, 0};

// Verify that a new material is vacuum everywhere.
TEST_CASE("Material starts as vacuum", "[unit][material]")
{
    const Material mat(Grid(7, 5));
    REQUIRE(mat.eps_r().size() == mat.grid().size());

    for (std::size_t k = 0; k < mat.grid().size(); ++k) {
        CHECK(mat.eps_r()[k] == fdtd::FloatType(1));
        CHECK(mat.mu_r()[k] == fdtd::FloatType(1));
        CHECK(mat.sigma_e()[k] == fdtd::FloatType(0));
        CHECK(mat.sigma_m()[k] == fdtd::FloatType(0));
    }
}

// Verify that fill_rect sets exactly the half-open rectangle.
TEST_CASE("Material fill_rect sets the half-open rectangle", "[unit][material]")
{
    Material mat(Grid(8, 6));
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
TEST_CASE("Material fill_rect clips to the grid", "[unit][material]")
{
    Material mat(Grid(4, 4));
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
TEST_CASE("Material fill sets every cell", "[unit][material]")
{
    Material mat(Grid(3, 2));
    mat.fill(GLASS);

    for (std::size_t k = 0; k < mat.grid().size(); ++k) {
        CHECK(mat.eps_r()[k] == GLASS.eps_r);
        CHECK(mat.sigma_e()[k] == GLASS.sigma_e);
    }
}

// Verify fill_circle against the cell-center test, including a center off the grid.
TEST_CASE("Material fill_circle sets the cells within the radius", "[unit][material]")
{
    const double ci = GENERATE(4.0, 4.5, -1.0, 8.5);
    const double cj = 3.0;
    const double r = 2.5;
    Material mat(Grid(9, 7));
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
TEST_CASE("Material fill_circle with zero radius sets one cell", "[unit][material]")
{
    Material mat(Grid(5, 5));
    mat.fill_circle(2, 3, 0, GLASS);

    std::size_t filled = 0;
    for (std::size_t k = 0; k < mat.grid().size(); ++k) {
        filled += mat.eps_r()[k] == GLASS.eps_r;
    }
    CHECK(filled == 1);
    CHECK(mat.at(2, 3).eps_r == GLASS.eps_r);
}

// Verify that unphysical media, bad circles and out-of-range cells are rejected.
TEST_CASE("Material rejects invalid input", "[unit][material]")
{
    Material mat(Grid(4, 4));
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

// Tolerance of a profile value against its double formula.
static const double PML_REL = sizeof(fdtd::FloatType) == sizeof(float) ? 1e-6 : 1e-12;

// Conductivity at the walls for a layer of 8 cells, order 3 and R0 = 1e-6.
static double pml_max(double cellSize)
{
    return -4 * std::log(1e-6) / (2 * fdtd::ETA_0 * 8 * cellSize);
}

// Verify that a new material has no PML and that the media leave the profiles alone.
TEST_CASE("Material starts without a PML", "[unit][material][pml]")
{
    Material mat(Grid(7, 5));
    mat.fill(GLASS);
    CHECK(mat.pml_thickness() == 0);
    REQUIRE(mat.sigma_e_x().size() == mat.grid().size());
    for (std::size_t k = 0; k < mat.grid().size(); ++k) {
        CHECK(mat.sigma_e_x()[k] == fdtd::FloatType(0));
        CHECK(mat.sigma_e_y()[k] == fdtd::FloatType(0));
        CHECK(mat.sigma_m_x()[k] == fdtd::FloatType(0));
        CHECK(mat.sigma_m_y()[k] == fdtd::FloatType(0));
    }
}

// Verify the PML checks, that a rejected call changes nothing, and that thickness 0 is off.
TEST_CASE("Material set_pml rejects bad settings", "[unit][material][pml]")
{
    Material mat(Grid(21, 21));
    CHECK_NOTHROW(mat.set_pml(10, 3, 1e-6));
    CHECK_THROWS_AS(Material(Grid(20, 21)).set_pml(10, 3, 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(Material(Grid(21, 20)).set_pml(10, 3, 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(mat.set_pml(4, 0, 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(mat.set_pml(4, std::numeric_limits<double>::infinity(), 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(mat.set_pml(4, std::numeric_limits<double>::quiet_NaN(), 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(mat.set_pml(4, 3, 0), std::invalid_argument);
    CHECK_THROWS_AS(mat.set_pml(4, 3, 1), std::invalid_argument);
    CHECK_THROWS_AS(mat.set_pml(4, 3, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    CHECK(mat.pml_thickness() == 10);
    CHECK(mat.sigma_e_x()[mat.grid().index(0, 10)] > fdtd::FloatType(0));

    CHECK_NOTHROW(mat.set_pml(0, -1, 2));
    CHECK(mat.pml_thickness() == 0);
    for (std::size_t k = 0; k < mat.grid().size(); ++k) {
        CHECK(mat.sigma_e_x()[k] == fdtd::FloatType(0));
        CHECK(mat.sigma_m_y()[k] == fdtd::FloatType(0));
    }
}

// Verify the electric profiles: sigma_max at the walls, graded inside, zero past, mirrored.
TEST_CASE("Material set_pml grades the electric profiles", "[unit][material][pml]")
{
    const Grid grid(40, 30, fdtd::FloatType(1e-3), fdtd::FloatType(2e-3));
    Material mat(grid);
    mat.set_pml(8, 3, 1e-6);
    const double maxX = pml_max(double(grid.dx()));
    const double maxY = pml_max(double(grid.dy()));
    const auto ex = [&](std::size_t i, std::size_t j) {
        return mat.sigma_e_x()[grid.index(i, j)];
    };
    const auto ey = [&](std::size_t i, std::size_t j) {
        return mat.sigma_e_y()[grid.index(i, j)];
    };

    CHECK(mat.pml_thickness() == 8);
    CHECK_THAT(double(ex(0, 15)), WithinRel(maxX, PML_REL));
    CHECK_THAT(double(ex(39, 15)), WithinRel(maxX, PML_REL));
    CHECK_THAT(double(ex(4, 15)), WithinRel(maxX / 8, PML_REL));
    CHECK(ex(8, 15) == fdtd::FloatType(0));
    CHECK(ex(20, 15) == fdtd::FloatType(0));
    CHECK(ex(31, 15) == fdtd::FloatType(0));
    CHECK_THAT(double(ey(20, 0)), WithinRel(maxY, PML_REL));
    CHECK_THAT(double(ey(20, 29)), WithinRel(maxY, PML_REL));
    CHECK(ey(20, 8) == fdtd::FloatType(0));
    CHECK(ey(20, 21) == fdtd::FloatType(0));

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(ex(i, 15) > ex(i + 1, 15));
        CHECK(ex(i, 15) == ex(39 - i, 15));
        CHECK(ey(20, i) > ey(20, i + 1));
        CHECK(ey(20, i) == ey(20, 29 - i));
    }
    for (std::size_t j = 0; j < 30; ++j) {
        CHECK(ex(3, j) == ex(3, 0));
    }
    for (std::size_t i = 0; i < 40; ++i) {
        CHECK(ey(i, 3) == ey(0, 3));
    }
}

// Verify that the normal-incidence round trip gives back the target reflection.
TEST_CASE("Material set_pml reaches the target reflection", "[unit][material][pml]")
{
    const Grid grid(40, 30, fdtd::FloatType(1e-3), fdtd::FloatType(1e-3));
    Material mat(grid);
    mat.set_pml(8, 3, 1e-6);
    const double sigmaMax = double(mat.sigma_e_x()[grid.index(0, 15)]);
    const double integral = sigmaMax * 8 * double(grid.dx()) / 4;
    CHECK_THAT(std::exp(-2 * fdtd::ETA_0 * integral), WithinRel(1e-6, 1e-4));
}

// Verify that the magnetic profiles sit half a cell further, scaled by mu0 / eps0.
TEST_CASE("Material set_pml grades the magnetic profiles", "[unit][material][pml]")
{
    const Grid grid(40, 30, fdtd::FloatType(1e-3), fdtd::FloatType(2e-3));
    Material mat(grid);
    mat.set_pml(8, 3, 1e-6);
    const double match = fdtd::MU_0 / fdtd::EPS_0;
    const double maxX = pml_max(double(grid.dx()));
    const double maxY = pml_max(double(grid.dy()));
    const auto mx = [&](std::size_t i, std::size_t j) {
        return mat.sigma_m_x()[grid.index(i, j)];
    };
    const auto my = [&](std::size_t i, std::size_t j) {
        return mat.sigma_m_y()[grid.index(i, j)];
    };

    CHECK_THAT(double(mx(7, 15)), WithinRel(match * maxX / 4096, PML_REL));
    CHECK_THAT(double(mx(31, 15)), WithinRel(match * maxX / 4096, PML_REL));
    CHECK_THAT(double(mx(0, 15)), WithinRel(match * maxX * std::pow(7.5 / 8, 3), PML_REL));
    CHECK(mx(8, 15) == fdtd::FloatType(0));
    CHECK(mx(30, 15) == fdtd::FloatType(0));
    CHECK_THAT(double(my(20, 7)), WithinRel(match * maxY / 4096, PML_REL));
    CHECK_THAT(double(my(20, 21)), WithinRel(match * maxY / 4096, PML_REL));
    CHECK(my(20, 8) == fdtd::FloatType(0));
    CHECK(my(20, 20) == fdtd::FloatType(0));

    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(mx(i, 15) == mx(38 - i, 15));
        CHECK(my(20, i) == my(20, 28 - i));
    }
}
