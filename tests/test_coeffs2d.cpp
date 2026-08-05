/*
 * C++ file for the 2D TMz update coefficient tests.
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
#include "fdtd/TMz/fdtd_tmz_coeffs2d.h"
#include "fdtd/TMz/fdtd_tmz_courant.h"
#include "fdtd/fdtd_constants.h"

// Standard library for limits and errors.
#include <cstddef>
#include <limits>
#include <stdexcept>

// Shorthand for the types under test.
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fdtd::TMz::Coeffs2D;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Grid2D;
using fdtd::TMz::Material2D;

// Relative tolerance that holds in single precision.
static constexpr double TOL = 1e-6;

// Verify that vacuum gives no decay and curl factors dt/EPS_0 and dt/MU_0.
TEST_CASE("Coeffs2D in vacuum", "[unit][coeffs2d]")
{
    const Grid2D grid(5, 4, 1e-3F, 2e-3F);
    const Material2D material(grid);
    const double dt = courant_dt(grid, 0.9);
    const Coeffs2D coeffs(material, dt);

    REQUIRE(coeffs.grid().nx() == 5);
    REQUIRE(coeffs.grid().ny() == 4);
    CHECK(coeffs.dt() == dt);
    REQUIRE(coeffs.ce1().size() == grid.size());
    REQUIRE(coeffs.ce2().size() == grid.size());
    REQUIRE(coeffs.ch1().size() == grid.size());
    REQUIRE(coeffs.ch2().size() == grid.size());

    for (std::size_t k = 0; k < grid.size(); ++k) {
        CHECK(coeffs.ce1()[k] == fdtd::FloatType(1));
        CHECK(coeffs.ch1()[k] == fdtd::FloatType(1));
        CHECK_THAT(double(coeffs.ce2()[k]), WithinRel(dt / fdtd::EPS_0, TOL));
        CHECK_THAT(double(coeffs.ch2()[k]), WithinRel(dt / fdtd::MU_0, TOL));
    }
}

// Verify a lossy dielectric and magnetic cell against hand values, and that the
// cells around it stay vacuum.
TEST_CASE("Coeffs2D in a lossy medium", "[unit][coeffs2d]")
{
    const Grid2D grid(3, 3);
    Material2D material(grid);
    const Material2D::Medium lossy{4, 2, 0.5, 300};
    material.fill_rect(1, 1, 2, 2, lossy);
    const double dt = 1e-12;
    const Coeffs2D coeffs(material, dt);

    const double eps = 4 * fdtd::EPS_0;
    const double mu = 2 * fdtd::MU_0;
    const double a = 0.5 * dt / (2 * eps);
    const double b = 300 * dt / (2 * mu);
    const std::size_t k = grid.index(1, 1);

    CHECK_THAT(double(coeffs.ce1()[k]), WithinRel((1 - a) / (1 + a), TOL));
    CHECK_THAT(double(coeffs.ce2()[k]), WithinRel(dt / eps / (1 + a), TOL));
    CHECK_THAT(double(coeffs.ch1()[k]), WithinRel((1 - b) / (1 + b), TOL));
    CHECK_THAT(double(coeffs.ch2()[k]), WithinRel(dt / mu / (1 + b), TOL));
    CHECK(coeffs.ce1()[k] < fdtd::FloatType(1));
    CHECK(coeffs.ch1()[k] < fdtd::FloatType(1));

    const std::size_t v = grid.index(0, 0);
    CHECK(coeffs.ce1()[v] == fdtd::FloatType(1));
    CHECK_THAT(double(coeffs.ce2()[v]), WithinRel(dt / fdtd::EPS_0, TOL));
}

// Verify that a perfect-conductor limit drives Ce1 to -1 and Ce2 far below vacuum.
TEST_CASE("Coeffs2D approaches a conductor", "[unit][coeffs2d]")
{
    const Grid2D grid(1, 1);
    Material2D material(grid);
    material.fill({1, 1, 1e12F, 0});
    const Coeffs2D coeffs(material, 1e-12);

    CHECK_THAT(double(coeffs.ce1()[0]), WithinAbs(-1.0, 1e-6));
    CHECK(double(coeffs.ce2()[0]) < 1e-9 * (1e-12 / fdtd::EPS_0));
}

// Verify that time steps that are not positive and finite are rejected.
TEST_CASE("Coeffs2D rejects invalid time steps", "[unit][coeffs2d]")
{
    const Material2D material(Grid2D(2, 2));

    CHECK_THROWS_AS(Coeffs2D(material, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(Coeffs2D(material, -1e-12), std::invalid_argument);
    CHECK_THROWS_AS(Coeffs2D(material, std::numeric_limits<double>::infinity()),
                    std::invalid_argument);
    CHECK_THROWS_AS(Coeffs2D(material, std::numeric_limits<double>::quiet_NaN()),
                    std::invalid_argument);
}
