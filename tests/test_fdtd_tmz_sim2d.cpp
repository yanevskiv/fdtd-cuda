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
#include "fdtd/TMz/fdtd_tmz_grid2d.h"
#include "fdtd/TMz/fdtd_tmz_sim2d.h"

// Standard library for limits and errors.
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

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
    CHECK(opts.pml_thickness == 20);
    CHECK(opts.pml_order == 3);
    CHECK(opts.pml_reflection == 1e-6);
    CHECK(sim.pml_thickness() == 0);
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

// Verify that the PML is active only with Boundary::Pml and a nonzero thickness.
TEST_CASE("Sim2D activates the PML only with Boundary::Pml", "[unit][sim2d][pml]")
{
    Sim2D::Options opts;
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 5;
    CHECK(Sim2D(16, 12, opts).pml_thickness() == 5);
    CHECK(Sim2D(16, 12, opts).backend().boundary() == Boundary::Pml);
    CHECK(Sim2D(16, 12, opts).material().pml_thickness() == 5);

    opts.pml_thickness = 0;
    CHECK(Sim2D(16, 12, opts).pml_thickness() == 0);

    opts.boundary = Boundary::Pec;
    opts.pml_thickness = 5;
    CHECK(Sim2D(16, 12, opts).pml_thickness() == 0);

    opts.boundary = Boundary::Pmc;
    CHECK(Sim2D(16, 12, opts).pml_thickness() == 0);
}

// Verify that bad PML settings throw only when the PML is active.
TEST_CASE("Sim2D checks the PML settings when active", "[unit][sim2d][pml]")
{
    Sim2D::Options opts;
    opts.pml_thickness = 6;
    opts.pml_order = -1;
    opts.pml_reflection = 2;
    CHECK_NOTHROW(Sim2D(12, 12, opts));

    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 0;
    CHECK_NOTHROW(Sim2D(12, 12, opts));

    opts = {};
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 6;
    CHECK_THROWS_AS(Sim2D(12, 20, opts), std::invalid_argument);
    CHECK_THROWS_AS(Sim2D(20, 12, opts), std::invalid_argument);
    CHECK_NOTHROW(Sim2D(13, 13, opts));

    opts.pml_order = 0;
    CHECK_THROWS_AS(Sim2D(16, 16, opts), std::invalid_argument);

    opts = {};
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 4;
    opts.pml_reflection = 1;
    CHECK_THROWS_AS(Sim2D(16, 16, opts), std::invalid_argument);
}

// Verify that set_material grades the options' PML into the material, or clears it.
TEST_CASE("Sim2D set_material keeps the PML of the options", "[unit][sim2d][pml]")
{
    Sim2D::Options opts;
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 3;
    Sim2D sim(12, 12, opts);
    const Material2D vacuum(sim.grid());
    sim.set_material(vacuum);
    CHECK(sim.material().pml_thickness() == 3);
    CHECK(sim.material().sigma_e_x()[sim.grid().index(0, 6)] > FloatType(0));

    Sim2D pec(12, 12);
    Material2D graded(pec.grid());
    graded.set_pml(3, 3, 1e-6);
    pec.set_material(graded);
    CHECK(pec.pml_thickness() == 0);
    CHECK(pec.material().sigma_e_x()[pec.grid().index(0, 6)] == FloatType(0));
}

// Verify that the edge Ez takes no source with the PML, as under PEC.
TEST_CASE("Sim2D rejects an edge Ez source with the PML", "[unit][sim2d][pml]")
{
    Sim2D::Options opts;
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 2;
    Sim2D sim(8, 8, opts);
    CHECK_THROWS_AS(sim.add_source(0, 3, Component::Ez, ramp), std::invalid_argument);
}

// Verify that sources in the PML are rejected for every component, and taken just past it.
TEST_CASE("Sim2D rejects sources inside the PML", "[unit][sim2d][pml]")
{
    Sim2D::Options opts;
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 3;
    Sim2D sim(12, 12, opts);
    CHECK_THROWS_AS(sim.add_source(2, 6, Component::Ez, ramp), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_source(6, 2, Component::Ez, ramp), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_source(6, 9, Component::Hx, ramp), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_source(9, 6, Component::Hy, ramp), std::invalid_argument);
    CHECK_NOTHROW(sim.add_source(3, 3, Component::Ez, ramp));
    CHECK_NOTHROW(sim.add_source(8, 8, Component::Hy, ramp));
    CHECK(sim.source_count() == 2);

    Sim2D pec(12, 12);
    CHECK_NOTHROW(pec.add_source(2, 6, Component::Ez, ramp));
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
    CHECK_THROWS_AS(sim.add_source(2, 2, Component::Ez, fdtd::TMz::Waveform{}),
                    std::invalid_argument);
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
        sim->add_source(5, 6, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim->dt(), 4));
        sim->add_source(9, 3, Component::Hy, fdtd::TMz::Gaussian::from_steps(sim->dt(), 6));
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

// Verify that probes record at their Yee times and match the downloaded fields.
TEST_CASE("Sim2D probes record at their Yee times", "[unit][sim2d]")
{
    Sim2D sim(9, 9);
    sim.add_source(4, 4, Component::Ez, ramp);
    sim.add_source(1, 1, Component::Hx, ramp);
    const std::size_t ez = sim.add_probe(4, 4, Component::Ez);
    const std::size_t hx = sim.add_probe(1, 1, Component::Hx);
    const std::size_t hy = sim.add_probe(6, 2, Component::Hy);
    CHECK(sim.probe_count() == 3);
    CHECK(sim.probe_data(ez).empty());

    sim.run(1);
    const double dt = sim.dt();
    CHECK(sim.probe_data(ez) == std::vector<FloatType>{FloatType(dt)});
    CHECK(sim.probe_data(hx) == std::vector<FloatType>{FloatType(0.5 * dt)});
    CHECK(sim.probe_times(ez) == std::vector<double>{dt});
    CHECK(sim.probe_times(hx) == std::vector<double>{0.5 * dt});

    sim.run(4);
    sim.download();
    const auto data = sim.probe_data(hy);
    const auto times = sim.probe_times(hy);
    REQUIRE(data.size() == 5);
    REQUIRE(times.size() == 5);
    CHECK(data.back() == sim.fields().hy()[sim.grid().index(6, 2)]);
    CHECK(times.back() == 4.5 * dt);
}

// Verify that probe records from run(a) then run(b) match run(a + b) bit for bit.
TEST_CASE("Sim2D probe records compose", "[unit][sim2d]")
{
    Sim2D split(16, 12);
    Sim2D whole(16, 12);
    for (Sim2D *sim : {&split, &whole}) {
        sim->add_source(5, 6, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim->dt(), 4));
        sim->add_probe(10, 6, Component::Ez);
        sim->add_probe(10, 6, Component::Hy);
    }
    split.run(13);
    split.run(18);
    whole.run(31);
    for (std::size_t id = 0; id < 2; ++id) {
        CHECK(split.probe_data(id) == whole.probe_data(id));
        CHECK(split.probe_times(id) == whole.probe_times(id));
    }
    const auto ez = whole.probe_data(0);
    CHECK(std::any_of(ez.begin(), ez.end(), [](FloatType v) {
        return v != 0;
    }));
}

// Verify that bad probes throw: outside the grid, after a run, or an unknown id.
TEST_CASE("Sim2D checks probes", "[unit][sim2d]")
{
    Sim2D sim(8, 6);
    CHECK_THROWS_AS(sim.add_probe(8, 0, Component::Ez), std::invalid_argument);
    CHECK_THROWS_AS(sim.add_probe(0, 6, Component::Hx), std::invalid_argument);
    CHECK(sim.add_probe(7, 5, Component::Hy) == 0);
    CHECK_THROWS_AS(sim.probe_data(1), std::out_of_range);
    CHECK_THROWS_AS(sim.probe_times(1), std::out_of_range);

    sim.run(1);
    CHECK_THROWS_AS(sim.add_probe(2, 2, Component::Ez), std::logic_error);
    CHECK(sim.probe_count() == 1);
    CHECK(sim.probe_data(0) == std::vector<FloatType>{FloatType(0)});
}

// Verify that scheduled snapshots match the fields of a simulation stopped at the same step.
TEST_CASE("Sim2D takes scheduled snapshots", "[unit][sim2d]")
{
    Sim2D sim(16, 12);
    Sim2D ref(16, 12);
    for (Sim2D *s : {&sim, &ref}) {
        s->add_source(5, 6, Component::Ez, fdtd::TMz::Gaussian::from_steps(s->dt(), 4));
    }
    sim.add_snapshot(20);
    sim.add_snapshot(7);
    sim.add_snapshot(0);
    sim.add_snapshot(7);
    CHECK(sim.snapshots().size() == 1);
    sim.run(10);
    sim.run(15);
    REQUIRE(sim.snapshots().size() == 3);
    CHECK(sim.step_count() == 25);

    const std::size_t steps[] = {0, 7, 20};
    for (std::size_t k = 0; k < 3; ++k) {
        const auto &snap = sim.snapshots()[k];
        CHECK(snap.step == steps[k]);
        ref.run(steps[k] - ref.step_count());
        ref.download();
        const auto &a = snap.fields;
        const auto &b = ref.fields();
        CHECK(std::equal(a.ez().begin(), a.ez().end(), b.ez().begin()));
        CHECK(std::equal(a.hx().begin(), a.hx().end(), b.hx().begin()));
        CHECK(std::equal(a.hy().begin(), a.hy().end(), b.hy().begin()));
    }
    CHECK(std::any_of(sim.snapshots()[2].fields.ez().begin(), sim.snapshots()[2].fields.ez().end(),
                      [](FloatType v) {
                          return v != 0;
                      }));

    ref.run(5);
    ref.download();
    const auto last = sim.snapshot();
    CHECK(last.step == 25);
    CHECK(std::equal(last.fields.ez().begin(), last.fields.ez().end(), ref.fields().ez().begin()));
}

// Verify that a snapshot step already passed throws.
TEST_CASE("Sim2D checks snapshot steps", "[unit][sim2d]")
{
    Sim2D sim(8, 6);
    sim.run(3);
    CHECK_THROWS_AS(sim.add_snapshot(2), std::invalid_argument);
    sim.add_snapshot(3);
    REQUIRE(sim.snapshots().size() == 1);
    CHECK(sim.snapshots()[0].step == 3);
}
