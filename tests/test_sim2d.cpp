/*
 * C++ file for the 2D TMz simulation tests.
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
#include "fdtd/TMz/fdtd_tmz_sim2d.h"

// Standard library for limits and errors.
#include <limits>
#include <stdexcept>

// Shorthand for the types under test.
using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::TMz::BackendType;
using fdtd::TMz::Boundary;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Sim2D;

// Verify the defaults: millimeter cells, S = 0.99, PEC, serial, stability checked.
TEST_CASE("Sim2D defaults", "[unit][sim2d]")
{
    const Sim2D sim(16, 8);
    const Sim2D::Options opts;

    CHECK(sim.grid().nx() == 16);
    CHECK(sim.grid().ny() == 8);
    CHECK(sim.grid().dx() == opts.dx);
    CHECK(sim.grid().dy() == opts.dy);
    CHECK(opts.courant == 0.99);
    CHECK(opts.dt == 0);
    CHECK(opts.check_stability);
    CHECK_THAT(sim.dt(), WithinRel(courant_dt(sim.grid(), 0.99), 1e-12));
    CHECK(sim.backend().type() == BackendType::Serial);
    CHECK(sim.backend().boundary() == Boundary::Pec);
    CHECK(sim.material().grid().nx() == 16);
}

// Verify that the options reach the grid, the time step and the backend.
TEST_CASE("Sim2D applies its options", "[unit][sim2d]")
{
    Sim2D::Options opts;
    opts.dx = FloatType(2e-3);
    opts.dy = FloatType(1e-3);
    opts.courant = 0.5;
    opts.boundary = Boundary::Pmc;
    const Sim2D sim(8, 8, opts);

    CHECK(sim.grid().dx() == FloatType(2e-3));
    CHECK(sim.grid().dy() == FloatType(1e-3));
    CHECK_THAT(sim.dt(), WithinRel(courant_dt(sim.grid(), 0.5), 1e-12));
    CHECK(sim.backend().boundary() == Boundary::Pmc);
    CHECK(sim.options().courant == 0.5);
}

// Verify that an explicit dt overrides the Courant number.
TEST_CASE("Sim2D takes an explicit dt", "[unit][sim2d]")
{
    Sim2D::Options opts;
    opts.courant = 2;
    opts.dt = 1e-12;
    const Sim2D sim(8, 8, opts);

    CHECK(sim.dt() == 1e-12);
}

// Verify that bad options throw.
TEST_CASE("Sim2D rejects bad options", "[unit][sim2d]")
{
    CHECK_THROWS_AS(Sim2D(0, 8), std::invalid_argument);

    Sim2D::Options opts;
    opts.dx = 0;
    CHECK_THROWS_AS(Sim2D(8, 8, opts), std::invalid_argument);

    opts = {};
    opts.courant = 1.5;
    CHECK_THROWS_AS(Sim2D(8, 8, opts), std::invalid_argument);

    opts = {};
    opts.dt = -1e-12;
    CHECK_THROWS_AS(Sim2D(8, 8, opts), std::invalid_argument);

    opts = {};
    opts.dt = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(Sim2D(8, 8, opts), std::invalid_argument);

    opts = {};
    opts.dt = 1e-9;
    CHECK_THROWS_AS(Sim2D(8, 8, opts), std::invalid_argument);
}
