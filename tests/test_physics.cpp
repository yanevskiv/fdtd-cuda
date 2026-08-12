/*
 * C++ file for the physics tests.
 *
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_material.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"
#include "fdtd/fdtd_constants.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Component;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Fields;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;
using fdtd::TMz::Simulation;

namespace
{
    // Square interior, in cells, of the PML reflection runs.
    constexpr std::size_t REFLECTION_INTERIOR = 60;

    // Margin around the reference interior, wide enough that its wall echoes nothing in time.
    constexpr std::size_t REFLECTION_MARGIN = 100;

    // Steps of a reflection run.
    constexpr std::size_t REFLECTION_STEPS = 250;

    // A cell of the grid by column and row.
    struct Cell
    {
        std::size_t i;
        std::size_t j;
    };

    // Largest magnitude in a sequence and where it is.
    struct Peak
    {
        std::size_t index;
        double value;
    };

    // Check that two grids have the same size.
    void require_same_size(const Grid &a, const Grid &b)
    {
        if (a.nx() != b.nx() || a.ny() != b.ny()) {
            throw std::invalid_argument("field_energy: grids differ in size");
        }
    }

    // Sum the energy with Ez from one set of fields and the product of H from two.
    double energy(std::span<const FloatType> ez, const Fields &h0, const Fields &h1,
                  const Material &material)
    {
        const Grid &grid = material.grid();
        const auto epsR = material.eps_r();
        const auto muR = material.mu_r();
        double sum = 0;
        for (std::size_t j = 0; j < grid.ny(); ++j) {
            for (std::size_t i = 0; i < grid.nx(); ++i) {
                const std::size_t k = grid.index(i, j);
                const double e = double(ez[k]);
                const double hh = double(h0.hx()[k]) * double(h1.hx()[k]) +
                                  double(h0.hy()[k]) * double(h1.hy()[k]);
                sum += fdtd::EPS_0 * double(epsR[k]) * e * e + fdtd::MU_0 * double(muR[k]) * hh;
            }
        }
        return 0.5 * sum * double(grid.dx()) * double(grid.dy());
    }

    // Return the field energy per unit length in z, in J/m, summed over the cells.
    double field_energy(const Fields &fields, const Material &material)
    {
        require_same_size(fields.grid(), material.grid());
        return energy(fields.ez(), fields, fields, material);
    }

    // Return the energy Yee conserves, eps Ez^2 / 2 + mu H^(n-1/2) . H^(n+1/2) / 2, in J/m.
    double field_energy(const Fields &fields, const Fields &next, const Material &material)
    {
        require_same_size(fields.grid(), material.grid());
        require_same_size(next.grid(), material.grid());
        return energy(fields.ez(), fields, next, material);
    }

    // Return the largest magnitude in a sequence and where it is, the first on a tie.
    Peak find_peak(std::span<const FloatType> values)
    {
        if (values.empty()) {
            throw std::invalid_argument("find_peak: no values");
        }
        Peak peak{0, std::abs(double(values[0]))};
        for (std::size_t k = 1; k < values.size(); ++k) {
            const double magnitude = std::abs(double(values[k]));
            if (magnitude > peak.value) {
                peak = {k, magnitude};
            }
        }
        return peak;
    }

    // Add an Ez source and probes to an unrun simulation, run it and return one trace per probe.
    std::vector<std::vector<FloatType>> point_source_traces(Simulation &sim, Cell source,
                                                            fdtd::TMz::Waveform waveform,
                                                            std::span<const Cell> probes,
                                                            std::size_t nsteps)
    {
        sim.add_source(source.i, source.j, Component::Ez, std::move(waveform));
        std::vector<std::size_t> ids;
        ids.reserve(probes.size());
        for (const Cell &cell : probes) {
            ids.push_back(sim.add_probe(cell.i, cell.j, Component::Ez));
        }
        sim.run(nsteps);
        std::vector<std::vector<FloatType>> traces;
        traces.reserve(ids.size());
        for (const std::size_t id : ids) {
            traces.push_back(sim.probe_data(id));
        }
        return traces;
    }

    // Return the conserved energy now, running one extra step for the later H.
    double conserved_energy(Simulation &sim)
    {
        const auto now = sim.snapshot();
        sim.run(1);
        const auto next = sim.snapshot();
        return field_energy(now.fields, next.fields, sim.material());
    }

    // Run a centred Gaussian source and return the conserved energy after every block of steps.
    std::vector<double> energy_history(Simulation &sim, std::size_t blocks, std::size_t blockSteps)
    {
        const std::size_t n = sim.grid().nx();
        sim.add_source(n / 2, n / 2, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim.dt(), 10));
        std::vector<double> history;
        for (std::size_t b = 0; b < blocks; ++b) {
            sim.run(blockSteps);
            history.push_back(conserved_energy(sim));
        }
        return history;
    }

    // Record Ez at a probe from a Gaussian source in an interior inside a border of cells.
    std::vector<FloatType> interior_trace(const Simulation::Options &opts, std::size_t border,
                                          Cell source, Cell probe)
    {
        const std::size_t n = REFLECTION_INTERIOR + 2 * border;
        Simulation sim(n, n, opts);
        const std::vector<Cell> probes{{border + probe.i, border + probe.j}};
        return point_source_traces(sim, {border + source.i, border + source.j},
                                   fdtd::TMz::Gaussian::from_steps(sim.dt(), 8), probes,
                                   REFLECTION_STEPS)[0];
    }

    // Return a PML's reflection in dB against a PEC domain too large to reflect in time.
    double pml_reflection_db(std::size_t thickness, Cell source, Cell probe)
    {
        Simulation::Options opts;
        opts.boundary = Boundary::Pml;
        opts.pml_thickness = thickness;
        const auto trace = interior_trace(opts, thickness, source, probe);
        const auto reference =
            interior_trace(Simulation::Options(), REFLECTION_MARGIN, source, probe);
        double error = 0;
        for (std::size_t k = 0; k < trace.size(); ++k) {
            error = std::max(error, std::abs(double(trace[k]) - double(reference[k])));
        }
        return 20 * std::log10(error / find_peak(reference).value);
    }

    // Check that two runs left the same Ez, Hx and Hy, bit for bit.
    bool same_fields(const Fields &a, const Fields &b)
    {
        const auto same = [](std::span<const FloatType> x, std::span<const FloatType> y) {
            return std::ranges::equal(x, y);
        };
        return same(a.ez(), b.ez()) && same(a.hx(), b.hx()) && same(a.hy(), b.hy());
    }

    // Return the step of a trace's peak, refined by a parabola through it and its neighbours.
    double peak_step(const std::vector<FloatType> &trace)
    {
        const std::size_t k = find_peak(trace).index;
        if (k == 0 || k + 1 == trace.size()) {
            return double(k);
        }
        const double a = double(trace[k - 1]);
        const double b = double(trace[k]);
        const double c = double(trace[k + 1]);
        return double(k) + 0.5 * (a - c) / (a - 2 * b + c);
    }
}

// Verify the energy of hand-set fields in a medium with eps_r and mu_r.
TEST_CASE("field_energy sums the cell energies", "[unit][helpers]")
{
    const Grid grid(4, 3, FloatType(0.5), FloatType(0.25));
    Material material(grid);
    material.fill({FloatType(2), FloatType(3), FloatType(0), FloatType(0)});
    Fields fields(grid);
    fields.ez()[grid.index(1, 1)] = FloatType(2);
    fields.hx()[grid.index(2, 0)] = FloatType(1);
    fields.hy()[grid.index(0, 2)] = FloatType(-1);

    const double area = 0.5 * 0.25;
    const double expected = 0.5 * area * (fdtd::EPS_0 * 2 * 4 + fdtd::MU_0 * 3 * (1 + 1));
    CHECK_THAT(field_energy(fields, material), WithinRel(expected, 1e-6));
}

// Verify that the two-step energy takes Ez from the first fields and multiplies H.
TEST_CASE("field_energy multiplies H over two half steps", "[unit][helpers]")
{
    const Grid grid(3, 3);
    const Material material(grid);
    Fields fields(grid);
    Fields next(grid);
    fields.ez()[grid.index(1, 1)] = FloatType(1);
    fields.hx()[grid.index(1, 1)] = FloatType(1);
    fields.hy()[grid.index(2, 1)] = FloatType(2);
    next.ez()[grid.index(1, 1)] = FloatType(100);
    next.hx()[grid.index(1, 1)] = FloatType(3);
    next.hy()[grid.index(2, 1)] = FloatType(-1);

    const double expected = 0.5 * (fdtd::EPS_0 * 1 + fdtd::MU_0 * (3 - 2));
    CHECK_THAT(field_energy(fields, next, material), WithinRel(expected, 1e-6));
    CHECK_THROWS_AS(field_energy(Fields(Grid(2, 3)), material), std::invalid_argument);
}

// Verify that find_peak finds the first largest magnitude.
TEST_CASE("find_peak finds the largest magnitude", "[unit][helpers]")
{
    const std::vector<FloatType> values{FloatType(1), FloatType(-4), FloatType(3), FloatType(4)};
    const auto peak = find_peak(values);
    CHECK(peak.index == 1);
    CHECK(peak.value == 4);
    CHECK_THROWS_AS(find_peak({}), std::invalid_argument);
}

// Verify that point_source_traces returns one full trace per probe.
TEST_CASE("point_source_traces records every probe", "[unit][helpers]")
{
    Simulation sim(16, 16);
    const std::vector<Cell> probes{{8, 8}, {12, 8}};
    const auto traces =
        point_source_traces(sim, {8, 8}, fdtd::TMz::Gaussian::from_steps(sim.dt(), 4), probes, 20);
    REQUIRE(traces.size() == 2);
    CHECK(traces[0].size() == 20);
    CHECK(traces[1].size() == 20);
    CHECK(sim.step_count() == 20);
    CHECK(find_peak(traces[0]).value > find_peak(traces[1]).value);
}

// Verify that S = 0.99 keeps the energy of a PEC box bounded once the source is off.
TEST_CASE("S = 0.99 stays bounded", "[physics][stability]")
{
    Simulation::Options opts;
    opts.courant = 0.99;
    Simulation sim(64, 64, opts);
    const auto history = energy_history(sim, 20, 100);
    for (const double e : history) {
        const double ratio = e / history.front();
        CAPTURE(ratio);
        REQUIRE(std::isfinite(ratio));
        CHECK(ratio < 1.01);
        CHECK(ratio > 0.99);
    }
}

// Verify that S = 1.01 with the stability check off grows without bound.
TEST_CASE("S = 1.01 grows with check_stability off", "[physics][stability]")
{
    Simulation::Options opts;
    opts.check_stability = false;
    opts.dt = 1.01 * courant_dt(Grid(64, 64, opts.dx, opts.dy), 1.0);
    Simulation sim(64, 64, opts);
    const auto history = energy_history(sim, 20, 100);
    const double growth = history.back() / history.front();
    CAPTURE(growth);
    CHECK((!std::isfinite(growth) || growth > 1e6));
}

// Verify the speed C_0 within 1% along an axis and the diagonal, the axis being slower.
TEST_CASE("Pulses travel at the speed of light", "[physics][speed]")
{
    Simulation sim(256, 256);
    const std::vector<Cell> probes{{168, 128}, {218, 128}, {156, 156}, {191, 191}};
    const auto traces = point_source_traces(
        sim, {128, 128}, fdtd::TMz::Gaussian::from_steps(sim.dt(), 8), probes, 220);
    const double cellSteps = double(sim.grid().dx()) / (fdtd::C_0 * sim.dt());

    const double axial = peak_step(traces[1]) - peak_step(traces[0]);
    const double diagonal = peak_step(traces[3]) - peak_step(traces[2]);
    const double axialRatio = axial / (50 * cellSteps);
    const double diagonalRatio = diagonal / (35 * std::sqrt(2.0) * cellSteps);
    CAPTURE(axialRatio, diagonalRatio);
    CHECK(std::abs(axialRatio - 1) < 0.01);
    CHECK(std::abs(diagonalRatio - 1) < 0.01);
    CHECK(axialRatio > diagonalRatio);
}

// Verify that a cylindrical wave's peak falls off as 1 / sqrt(r) in the far field within 3%.
TEST_CASE("Far-field peak follows 1 / sqrt(r)", "[physics][falloff]")
{
    Simulation sim(256, 256);
    const std::size_t distances[] = {30, 50, 70, 90, 110};
    std::vector<Cell> probes;
    for (const std::size_t r : distances) {
        probes.push_back({128 + r, 128});
    }
    const auto traces = point_source_traces(
        sim, {128, 128}, fdtd::TMz::Gaussian::from_steps(sim.dt(), 8), probes, 240);

    const double reference = find_peak(traces[0]).value * std::sqrt(30.0);
    for (std::size_t k = 1; k < probes.size(); ++k) {
        const double ratio =
            find_peak(traces[k]).value * std::sqrt(double(distances[k])) / reference;
        CAPTURE(distances[k], ratio);
        CHECK(std::abs(ratio - 1) < 0.03);
    }
}

// Verify that a closed box with a rod conserves Yee's energy over 2000 steps, PEC and PMC.
TEST_CASE("Closed box conserves energy", "[physics][energy]")
{
    const Boundary boundary = GENERATE(Boundary::Pec, Boundary::Pmc);
    Simulation::Options opts;
    opts.boundary = boundary;
    Simulation sim(128, 128, opts);
    Material material(sim.grid());
    material.fill_circle(80, 70, 15, {FloatType(4), FloatType(1), FloatType(0), FloatType(0)});
    sim.set_material(material);
    sim.add_source(40, 50, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim.dt(), 8));

    std::vector<double> history;
    for (std::size_t b = 0; b < 20; ++b) {
        sim.run(b == 0 ? 100 : 99);
        const auto now = sim.snapshot();
        sim.run(1);
        const auto next = sim.snapshot();
        history.push_back(field_energy(now.fields, next.fields, sim.material()));
    }
    const double bound = 1000 * double(std::numeric_limits<FloatType>::epsilon());
    double drift = 0;
    for (const double e : history) {
        drift = std::max(drift, std::abs(e / history.front() - 1));
    }
    CAPTURE(boundary == Boundary::Pec, drift, bound);
    CHECK(history.front() > 0);
    CHECK(drift < bound);
}

// Verify bit for bit that the PML settings change nothing while the PML is off.
TEST_CASE("PML off reproduces the closed box bit for bit", "[physics][pml]")
{
    const Boundary boundary = GENERATE(Boundary::Pml, Boundary::Pec, Boundary::Pmc);
    Simulation::Options opts;
    opts.boundary = boundary;
    opts.pml_thickness = boundary == Boundary::Pml ? 0 : 12;
    opts.pml_order = 2;
    opts.pml_reflection = 1e-3;
    Simulation::Options plain;
    plain.boundary = boundary == Boundary::Pmc ? Boundary::Pmc : Boundary::Pec;

    const auto run = [](const Simulation::Options &options) {
        Simulation sim(48, 40, options);
        Material material(sim.grid());
        material.fill_circle(30, 22, 8, {FloatType(3), FloatType(1), FloatType(2), FloatType(0)});
        sim.set_material(material);
        sim.add_source(10, 12, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim.dt(), 8));
        sim.run(150);
        return sim.snapshot().fields;
    };
    CAPTURE(int(boundary));
    CHECK(same_fields(run(opts), run(plain)));
}

// Verify that the PML reflects below its thresholds at 10 and 20 cells and four incidences.
TEST_CASE("PML reflects below the threshold", "[physics][pml]")
{
    struct Case
    {
        const char *name;
        Cell source;
        Cell probe;
        std::size_t thickness;
        double threshold;
    };
    const Case c = GENERATE(
        Case{"normal", {20, 30}, {5, 30}, 10, -80}, Case{"normal", {20, 30}, {5, 30}, 20, -90},
        Case{"45 degrees", {10, 15}, {10, 35}, 10, -80},
        Case{"45 degrees", {10, 15}, {10, 35}, 20, -90}, Case{"grazing", {10, 3}, {50, 3}, 10, -60},
        Case{"grazing", {10, 3}, {50, 3}, 20, -80}, Case{"corner", {15, 15}, {5, 5}, 10, -75},
        Case{"corner", {15, 15}, {5, 5}, 20, -85});
    const double reflection = pml_reflection_db(c.thickness, c.source, c.probe);
    CAPTURE(c.name, c.thickness, reflection, c.threshold);
    CHECK(reflection < c.threshold);
}

// Verify that a PML box loses five orders of magnitude of energy in a thousand steps.
TEST_CASE("PML box loses its energy", "[physics][pml]")
{
    Simulation::Options opts;
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 10;
    Simulation sim(64, 64, opts);
    sim.add_source(32, 32, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim.dt(), 10));
    sim.run(60);
    const double start = conserved_energy(sim);
    sim.run(939);
    const double ratio = conserved_energy(sim) / start;
    CAPTURE(start, ratio);
    CHECK(start > 0);
    CHECK(ratio < 1e-5);
}

// Verify that a PML box's energy keeps falling over three thousand steps, with no late growth.
TEST_CASE("PML box does not grow", "[physics][pml][stability]")
{
    Simulation::Options opts;
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 10;
    Simulation sim(64, 64, opts);
    const auto history = energy_history(sim, 30, 99);
    for (std::size_t b = 1; b < history.size(); ++b) {
        const double ratio = history[b] / history[b - 1];
        CAPTURE(b, ratio);
        REQUIRE(std::isfinite(ratio));
        CHECK(ratio < 1);
    }
}
