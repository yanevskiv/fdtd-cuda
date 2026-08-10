/*
 * C++ file for the 2D TMz split-field PML profile tests.
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
#include "fdtd/TMz/fdtd_tmz_pml2d.h"
#include "fdtd/fdtd_constants.h"

// Standard library for math, sizes and errors.
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

// Shorthand for the functions under test.
using Catch::Matchers::WithinRel;
using fdtd::TMz::check_pml;
using fdtd::TMz::pml_depth;
using fdtd::TMz::pml_magnetic_sigma;
using fdtd::TMz::pml_sigma;
using fdtd::TMz::pml_sigma_max;

// Verify that a layer must fit twice and that order and reflection are in range.
TEST_CASE("check_pml limits", "[unit][pml]")
{
    CHECK_NOTHROW(check_pml(21, 21, 10, 3, 1e-6));
    CHECK_THROWS_AS(check_pml(20, 21, 10, 3, 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(check_pml(21, 20, 10, 3, 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(check_pml(21, 21, 10, 0, 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(check_pml(21, 21, 10, std::numeric_limits<double>::infinity(), 1e-6),
                    std::invalid_argument);
    CHECK_THROWS_AS(check_pml(21, 21, 10, std::nan(""), 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(check_pml(21, 21, 10, 3, 0), std::invalid_argument);
    CHECK_THROWS_AS(check_pml(21, 21, 10, 3, 1), std::invalid_argument);
    CHECK_THROWS_AS(check_pml(21, 21, 10, 3, std::nan("")), std::invalid_argument);
}

// Verify sigma_max against its formula and that the normal-incidence round trip,
// exp(-2 eta0 integral of sigma), gives back the target reflection.
TEST_CASE("pml_sigma_max reaches the target reflection", "[unit][pml]")
{
    const std::size_t thickness = 10;
    const double dx = 1e-3;
    const double sigmaMax = pml_sigma_max(thickness, dx, 3, 1e-6);
    CHECK_THAT(sigmaMax, WithinRel(-4 * std::log(1e-6) / (2 * fdtd::ETA_0 * 1e-2), 1e-12));

    const double integral = sigmaMax * double(thickness) * dx / 4;
    CHECK_THAT(std::exp(-2 * fdtd::ETA_0 * integral), WithinRel(1e-6, 1e-9));
}

// Verify the depth on both sides, at whole and half positions, and zero inside.
TEST_CASE("pml_depth on one axis", "[unit][pml]")
{
    const std::size_t n = 30;
    const std::size_t d = 5;
    CHECK(pml_depth(0, n, d) == 5);
    CHECK(pml_depth(29, n, d) == 5);
    CHECK(pml_depth(0.5, n, d) == 4.5);
    CHECK(pml_depth(28.5, n, d) == 4.5);
    CHECK(pml_depth(5, n, d) == 0);
    CHECK(pml_depth(24, n, d) == 0);
    CHECK(pml_depth(15, n, d) == 0);
    CHECK(pml_depth(3, n, 0) == 0);
}

// Verify the grading: zero inside, sigma_max at the walls, monotonic in between, mirrored.
TEST_CASE("pml_sigma grading", "[unit][pml]")
{
    const std::size_t n = 40;
    const std::size_t d = 8;
    const double sigmaMax = 2.5;
    CHECK(pml_sigma(8, n, d, 3, sigmaMax) == 0);
    CHECK(pml_sigma(20, n, d, 3, sigmaMax) == 0);
    CHECK_THAT(pml_sigma(0, n, d, 3, sigmaMax), WithinRel(sigmaMax, 1e-15));
    CHECK_THAT(pml_sigma(39, n, d, 3, sigmaMax), WithinRel(sigmaMax, 1e-15));
    CHECK_THAT(pml_sigma(4, n, d, 3, sigmaMax), WithinRel(sigmaMax / 8, 1e-15));
    CHECK_THAT(pml_sigma(6.5, n, d, 2, sigmaMax),
               WithinRel(sigmaMax * (1.5 / 8) * (1.5 / 8), 1e-15));
    for (double x = 0; x < 8; x += 0.5) {
        CHECK(pml_sigma(x, n, d, 3, sigmaMax) > pml_sigma(x + 0.5, n, d, 3, sigmaMax));
        CHECK(pml_sigma(x, n, d, 3, sigmaMax) == pml_sigma(39 - x, n, d, 3, sigmaMax));
    }
    CHECK(pml_sigma(0, n, 0, 3, sigmaMax) == 0);
}

// Verify the impedance match: sigma* / mu0 equals sigma / eps0.
TEST_CASE("pml_magnetic_sigma matches the impedance", "[unit][pml]")
{
    const double sigma = 0.75;
    CHECK_THAT(pml_magnetic_sigma(sigma) / fdtd::MU_0, WithinRel(sigma / fdtd::EPS_0, 1e-14));
    CHECK_THAT(pml_magnetic_sigma(sigma), WithinRel(sigma * fdtd::ETA_0 * fdtd::ETA_0, 1e-12));
    CHECK(pml_magnetic_sigma(0) == 0);
}
