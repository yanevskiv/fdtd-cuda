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
#include <algorithm>
#include <limits>
#include <stdexcept>

// Shorthand for the types under test.
using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::TMz::BackendType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Component;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Grid2D;
using fdtd::TMz::Material2D;
using fdtd::TMz::Sim2D;

// A waveform that returns the time it is sampled at.
static FloatType ramp(FloatType t)
{
    return t;
}

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

// Verify that check_stability = false accepts an explicit dt above the Courant limit.
TEST_CASE("Sim2D accepts an unstable dt without the check", "[unit][sim2d]")
{
    Sim2D::Options opts;
    opts.dt = 1e-9;
    opts.check_stability = false;
    Sim2D sim(8, 8, opts);
    CHECK(sim.dt() == 1e-9);
    CHECK(sim.dt() > courant_dt(sim.grid(), 1.0));
}

// Verify that run() counts steps and time, and that download() fills the host fields.
TEST_CASE("Sim2D counts steps and time", "[unit][sim2d]")
{
    Sim2D sim(8, 8);
    CHECK(sim.step_count() == 0);
    CHECK(sim.time() == 0);

    sim.run(3);
    sim.run(4);
    CHECK(sim.step_count() == 7);
    CHECK_THAT(sim.time(), WithinRel(7 * sim.dt(), 1e-12));

    sim.download();
    CHECK(sim.fields().ez().size() == sim.grid().size());
}

// Verify that set_material() keeps a matching material and rejects another grid.
TEST_CASE("Sim2D replaces its material", "[unit][sim2d]")
{
    Sim2D sim(8, 8);
    Material2D mat(sim.grid());
    mat.fill({4, 1, 0, 0});
    sim.set_material(mat);
    CHECK(sim.material().eps_r()[0] == FloatType(4));

    const Material2D other(Grid2D(4, 8, sim.grid().dx(), sim.grid().dy()));
    CHECK_THROWS_AS(sim.set_material(other), std::invalid_argument);
}

// Verify that add_source() accepts cells that are updated and rejects the rest.
TEST_CASE("Sim2D checks source cells", "[unit][sim2d]")
{
    Sim2D sim(8, 6);
    CHECK_THROWS_AS(sim.add_source(8, 2, Component::Ez, ramp), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_source(2, 6, Component::Ez, ramp), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_source(0, 2, Component::Ez, ramp), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_source(2, 5, Component::Ez, ramp), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_source(2, 5, Component::Hx, ramp), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_source(7, 2, Component::Hy, ramp), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_source(2, 2, Component::Ez, fdtd::Waveform{}), std::invalid_argument);
    CHECK(sim.source_count() == 0);

    CHECK_NOTHROW(sim.add_source(2, 2, Component::Ez, ramp));
    CHECK_NOTHROW(sim.add_source(7, 0, Component::Hx, ramp));
    CHECK_NOTHROW(sim.add_source(0, 5, Component::Hy, ramp));
    CHECK(sim.source_count() == 3);

    Sim2D::Options opts;
    opts.boundary = Boundary::Pmc;
    Sim2D pmc(8, 6, opts);
    CHECK_NOTHROW(pmc.add_source(0, 0, Component::Ez, ramp));
}

// Verify the sampling times: Ez at (n + 1) * dt, H at (n + 1/2) * dt.
TEST_CASE("Sim2D samples sources at their Yee times", "[unit][sim2d]")
{
    Sim2D sim(9, 9);
    sim.add_source(4, 4, Component::Ez, ramp);
    sim.add_source(1, 1, Component::Hx, ramp);
    sim.add_source(7, 7, Component::Hy, ramp);
    sim.run(1);
    sim.download();

    const auto &grid = sim.grid();
    const double dt = sim.dt();
    CHECK(sim.fields().ez()[grid.index(4, 4)] == FloatType(dt));
    CHECK(sim.fields().hx()[grid.index(1, 1)] == FloatType(0.5 * dt));
    CHECK(sim.fields().hy()[grid.index(7, 7)] == FloatType(0.5 * dt));
}

// Verify that sources on the same cell add up.
TEST_CASE("Sim2D adds sources on the same cell", "[unit][sim2d]")
{
    Sim2D sim(9, 9);
    sim.add_source(4, 4, Component::Ez, [](FloatType) {
        return FloatType(1);
    });
    sim.add_source(4, 4, Component::Ez, [](FloatType) {
        return FloatType(2);
    });
    sim.run(1);
    sim.download();
    CHECK(sim.fields().ez()[sim.grid().index(4, 4)] == FloatType(3));
}

// Verify that run(a) then run(b) matches run(a + b) bit for bit with a pulse running.
TEST_CASE("Sim2D runs compose with sources", "[unit][sim2d]")
{
    Sim2D::Options opts;
    opts.boundary = Boundary::Pmc;
    Sim2D split(16, 12, opts);
    Sim2D whole(16, 12, opts);
    for (Sim2D *sim : {&split, &whole}) {
        sim->add_source(5, 6, Component::Ez, fdtd::Gaussian::from_steps(sim->dt(), 4));
        sim->add_source(9, 3, Component::Hy, fdtd::Gaussian::from_steps(sim->dt(), 6));
    }
    split.run(13);
    split.run(18);
    whole.run(31);
    split.download();
    whole.download();

    const auto &a = split.fields();
    const auto &b = whole.fields();
    CHECK(std::equal(a.ez().begin(), a.ez().end(), b.ez().begin()));
    CHECK(std::equal(a.hx().begin(), a.hx().end(), b.hx().begin()));
    CHECK(std::equal(a.hy().begin(), a.hy().end(), b.hy().begin()));
    CHECK(std::any_of(a.ez().begin(), a.ez().end(), [](FloatType v) {
        return v != 0;
    }));
}
