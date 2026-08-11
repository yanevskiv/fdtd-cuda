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
    CHECK(coeffs.pml_thickness() == 0);
    for (const auto array :
         {coeffs.coeff_ez_x1(), coeffs.coeff_ez_x2(), coeffs.coeff_ez_y1(), coeffs.coeff_ez_y2(),
          coeffs.coeff_hx_1(), coeffs.coeff_hx_2(), coeffs.coeff_hy_1(), coeffs.coeff_hy_2()}) {
        REQUIRE(array.size() == grid.size());
    }

    for (std::size_t k = 0; k < grid.size(); ++k) {
        CHECK(coeffs.coeff_ez_x1()[k] == fdtd::FloatType(1));
        CHECK(coeffs.coeff_hx_1()[k] == fdtd::FloatType(1));
        CHECK_THAT(double(coeffs.coeff_ez_x2()[k]), WithinRel(dt / fdtd::EPS_0, TOL));
        CHECK_THAT(double(coeffs.coeff_hx_2()[k]), WithinRel(dt / fdtd::MU_0, TOL));
    }
}

// Verify a lossy dielectric and magnetic cell against hand values, its neighbours vacuum.
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

    CHECK_THAT(double(coeffs.coeff_ez_x1()[k]), WithinRel((1 - a) / (1 + a), TOL));
    CHECK_THAT(double(coeffs.coeff_ez_x2()[k]), WithinRel(dt / eps / (1 + a), TOL));
    CHECK_THAT(double(coeffs.coeff_hx_1()[k]), WithinRel((1 - b) / (1 + b), TOL));
    CHECK_THAT(double(coeffs.coeff_hx_2()[k]), WithinRel(dt / mu / (1 + b), TOL));
    CHECK(coeffs.coeff_ez_x1()[k] < fdtd::FloatType(1));
    CHECK(coeffs.coeff_hx_1()[k] < fdtd::FloatType(1));

    const std::size_t v = grid.index(0, 0);
    CHECK(coeffs.coeff_ez_x1()[v] == fdtd::FloatType(1));
    CHECK_THAT(double(coeffs.coeff_ez_x2()[v]), WithinRel(dt / fdtd::EPS_0, TOL));
}

// Verify that without a PML the x and y copies equal the single pre-split pair bit for bit.
TEST_CASE("Coeffs2D without a PML matches the isotropic coefficients", "[unit][coeffs2d]")
{
    const Grid2D grid(6, 5);
    Material2D material(grid);
    material.fill_rect(1, 1, 4, 3, {4, 2, 0.5F, 300});
    material.fill_circle(3, 2, 1.5, {2.5F, 1, 0.01F, 0});
    const double dt = courant_dt(grid, 0.9);
    const Coeffs2D coeffs(material, dt);

    for (std::size_t k = 0; k < grid.size(); ++k) {
        const double eps = double(material.eps_r()[k]) * fdtd::EPS_0;
        const double mu = double(material.mu_r()[k]) * fdtd::MU_0;
        const double a = double(material.sigma_e()[k]) * dt / (2.0 * eps);
        const double b = double(material.sigma_m()[k]) * dt / (2.0 * mu);
        const auto ce1 = fdtd::FloatType((1.0 - a) / (1.0 + a));
        const auto ce2 = fdtd::FloatType((dt / eps) / (1.0 + a));
        const auto ch1 = fdtd::FloatType((1.0 - b) / (1.0 + b));
        const auto ch2 = fdtd::FloatType((dt / mu) / (1.0 + b));

        CHECK(coeffs.coeff_ez_x1()[k] == ce1);
        CHECK(coeffs.coeff_ez_x2()[k] == ce2);
        CHECK(coeffs.coeff_ez_y1()[k] == ce1);
        CHECK(coeffs.coeff_ez_y2()[k] == ce2);
        CHECK(coeffs.coeff_hx_1()[k] == ch1);
        CHECK(coeffs.coeff_hx_2()[k] == ch2);
        CHECK(coeffs.coeff_hy_1()[k] == ch1);
        CHECK(coeffs.coeff_hy_2()[k] == ch2);
    }
}

// Verify that each coefficient pair takes the PML profile of its own direction only.
TEST_CASE("Coeffs2D follows the PML profiles by direction", "[unit][coeffs2d][pml]")
{
    const Grid2D grid(20, 16);
    Material2D material(grid);
    material.set_pml(4, 3, 1e-6);
    const double dt = courant_dt(grid, 0.9);
    const Coeffs2D coeffs(material, dt);
    CHECK(coeffs.pml_thickness() == 4);

    for (std::size_t k = 0; k < grid.size(); ++k) {
        const double ax = double(material.sigma_e_x()[k]) * dt / (2.0 * fdtd::EPS_0);
        const double ay = double(material.sigma_e_y()[k]) * dt / (2.0 * fdtd::EPS_0);
        const double bx = double(material.sigma_m_x()[k]) * dt / (2.0 * fdtd::MU_0);
        const double by = double(material.sigma_m_y()[k]) * dt / (2.0 * fdtd::MU_0);
        CHECK_THAT(double(coeffs.coeff_ez_x1()[k]), WithinRel((1 - ax) / (1 + ax), TOL));
        CHECK_THAT(double(coeffs.coeff_ez_x2()[k]), WithinRel(dt / fdtd::EPS_0 / (1 + ax), TOL));
        CHECK_THAT(double(coeffs.coeff_ez_y1()[k]), WithinRel((1 - ay) / (1 + ay), TOL));
        CHECK_THAT(double(coeffs.coeff_ez_y2()[k]), WithinRel(dt / fdtd::EPS_0 / (1 + ay), TOL));
        CHECK_THAT(double(coeffs.coeff_hx_1()[k]), WithinRel((1 - by) / (1 + by), TOL));
        CHECK_THAT(double(coeffs.coeff_hx_2()[k]), WithinRel(dt / fdtd::MU_0 / (1 + by), TOL));
        CHECK_THAT(double(coeffs.coeff_hy_1()[k]), WithinRel((1 - bx) / (1 + bx), TOL));
        CHECK_THAT(double(coeffs.coeff_hy_2()[k]), WithinRel(dt / fdtd::MU_0 / (1 + bx), TOL));
    }

    const std::size_t left = grid.index(0, 8);
    CHECK(coeffs.coeff_ez_x1()[left] < fdtd::FloatType(1));
    CHECK(coeffs.coeff_hy_1()[left] < fdtd::FloatType(1));
    CHECK(coeffs.coeff_ez_y1()[left] == fdtd::FloatType(1));
    CHECK(coeffs.coeff_hx_1()[left] == fdtd::FloatType(1));

    const std::size_t bottom = grid.index(10, 0);
    CHECK(coeffs.coeff_ez_y1()[bottom] < fdtd::FloatType(1));
    CHECK(coeffs.coeff_hx_1()[bottom] < fdtd::FloatType(1));
    CHECK(coeffs.coeff_ez_x1()[bottom] == fdtd::FloatType(1));
    CHECK(coeffs.coeff_hy_1()[bottom] == fdtd::FloatType(1));

    const std::size_t corner = grid.index(0, 0);
    CHECK(coeffs.coeff_ez_x1()[corner] < fdtd::FloatType(1));
    CHECK(coeffs.coeff_ez_y1()[corner] < fdtd::FloatType(1));

    const std::size_t centre = grid.index(10, 8);
    CHECK(coeffs.coeff_ez_x1()[centre] == fdtd::FloatType(1));
    CHECK(coeffs.coeff_ez_y1()[centre] == fdtd::FloatType(1));
    CHECK(coeffs.coeff_hx_1()[centre] == fdtd::FloatType(1));
    CHECK(coeffs.coeff_hy_1()[centre] == fdtd::FloatType(1));
}

// Verify that the PML scales with eps_r and mu_r, leaving the decay factors those of vacuum.
TEST_CASE("Coeffs2D scales the PML by the medium", "[unit][coeffs2d][pml]")
{
    const Grid2D grid(20, 16);
    Material2D vacuum(grid);
    vacuum.set_pml(4, 3, 1e-6);
    Material2D medium = vacuum;
    medium.fill({4, 2, 0, 0});
    const double dt = courant_dt(grid, 0.5);
    const Coeffs2D inVacuum(vacuum, dt);
    const Coeffs2D inMedium(medium, dt);

    for (const std::size_t k : {grid.index(0, 8), grid.index(10, 0), grid.index(1, 1)}) {
        CHECK_THAT(double(inMedium.coeff_ez_x1()[k]),
                   WithinRel(double(inVacuum.coeff_ez_x1()[k]), TOL));
        CHECK_THAT(double(inMedium.coeff_ez_y1()[k]),
                   WithinRel(double(inVacuum.coeff_ez_y1()[k]), TOL));
        CHECK_THAT(double(inMedium.coeff_hx_1()[k]),
                   WithinRel(double(inVacuum.coeff_hx_1()[k]), TOL));
        CHECK_THAT(double(inMedium.coeff_hy_1()[k]),
                   WithinRel(double(inVacuum.coeff_hy_1()[k]), TOL));
        CHECK_THAT(double(inMedium.coeff_ez_x2()[k]),
                   WithinRel(double(inVacuum.coeff_ez_x2()[k]) / 4, TOL));
        CHECK_THAT(double(inMedium.coeff_ez_y2()[k]),
                   WithinRel(double(inVacuum.coeff_ez_y2()[k]) / 4, TOL));
        CHECK_THAT(double(inMedium.coeff_hx_2()[k]),
                   WithinRel(double(inVacuum.coeff_hx_2()[k]) / 2, TOL));
        CHECK_THAT(double(inMedium.coeff_hy_2()[k]),
                   WithinRel(double(inVacuum.coeff_hy_2()[k]) / 2, TOL));
    }
}

// Verify that a perfect-conductor limit drives the Ez decay to -1, its curl factor near zero.
TEST_CASE("Coeffs2D approaches a conductor", "[unit][coeffs2d]")
{
    const Grid2D grid(1, 1);
    Material2D material(grid);
    material.fill({1, 1, 1e12F, 0});
    const Coeffs2D coeffs(material, 1e-12);

    CHECK_THAT(double(coeffs.coeff_ez_x1()[0]), WithinAbs(-1.0, 1e-6));
    CHECK(double(coeffs.coeff_ez_x2()[0]) < 1e-9 * (1e-12 / fdtd::EPS_0));
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
