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

#include "fdtd/TMz/fdtd_tmz_courant.h"
#include "fdtd/TMz/fdtd_tmz_sim2d.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"
#include "fdtd/fdtd_constants.h"
#include "support/test_support_physics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

using fdtd::FloatType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Component;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Grid2D;
using fdtd::TMz::Material2D;
using fdtd::TMz::Sim2D;
using test::support::Cell;

namespace
{
    // Return the conserved energy of the fields as they are now, running one extra step for
    // H half a step later.
    double conserved_energy(Sim2D &sim)
    {
        const auto now = sim.snapshot();
        sim.run(1);
        const auto next = sim.snapshot();
        return test::support::field_energy(now.fields, next.fields, sim.material());
    }

    // Run a centred Gaussian source and return the conserved energy after every block of
    // steps; each measurement runs one extra step.
    std::vector<double> energy_history(Sim2D &sim, std::size_t blocks, std::size_t blockSteps)
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

    // Square interior, in cells, of the PML reflection runs.
    constexpr std::size_t REFLECTION_INTERIOR = 60;

    // Margin around the interior of the reference run, wide enough that its PEC wall
    // reflects nothing back to the probe within the window.
    constexpr std::size_t REFLECTION_MARGIN = 100;

    // Steps of a reflection run.
    constexpr std::size_t REFLECTION_STEPS = 250;

    // Record Ez at a probe from a Gaussian source on a square grid of the interior plus a
    // border of some cells on each side, with the cells given from the interior's corner.
    std::vector<FloatType> interior_trace(const Sim2D::Options &opts, std::size_t border,
                                          Cell source, Cell probe)
    {
        const std::size_t n = REFLECTION_INTERIOR + 2 * border;
        Sim2D sim(n, n, opts);
        const std::vector<Cell> probes{{border + probe.i, border + probe.j}};
        return test::support::point_source_traces(sim, {border + source.i, border + source.j},
                                                  fdtd::TMz::Gaussian::from_steps(sim.dt(), 8),
                                                  probes, REFLECTION_STEPS)[0];
    }

    // Return the reflection of a PML of some thickness in dB: the largest difference
    // between the trace in front of it and the trace of a PEC domain too large to reflect
    // in time, over the largest value of the latter.
    double pml_reflection_db(std::size_t thickness, Cell source, Cell probe)
    {
        Sim2D::Options opts;
        opts.boundary = Boundary::Pml;
        opts.pml_thickness = thickness;
        const auto trace = interior_trace(opts, thickness, source, probe);
        const auto reference = interior_trace(Sim2D::Options(), REFLECTION_MARGIN, source, probe);
        double error = 0;
        for (std::size_t k = 0; k < trace.size(); ++k) {
            error = std::max(error, std::abs(double(trace[k]) - double(reference[k])));
        }
        return 20 * std::log10(error / test::support::find_peak(reference).value);
    }

    // Check that two runs left the same Ez, Hx and Hy, bit for bit.
    bool same_fields(const fdtd::TMz::Fields2D &a, const fdtd::TMz::Fields2D &b)
    {
        const auto same = [](std::span<const FloatType> x, std::span<const FloatType> y) {
            return std::ranges::equal(x, y);
        };
        return same(a.ez(), b.ez()) && same(a.hx(), b.hx()) && same(a.hy(), b.hy());
    }

    // Return the step of a trace's peak, refined between steps by a parabola through the
    // largest sample and its neighbours.
    double peak_step(const std::vector<FloatType> &trace)
    {
        const std::size_t k = test::support::find_peak(trace).index;
        if (k == 0 || k + 1 == trace.size()) {
            return double(k);
        }
        const double a = double(trace[k - 1]);
        const double b = double(trace[k]);
        const double c = double(trace[k + 1]);
        return double(k) + 0.5 * (a - c) / (a - 2 * b + c);
    }
}

// Verify that S = 0.99 keeps the energy of a PEC box bounded once the source is off.
TEST_CASE("S = 0.99 stays bounded", "[physics][stability]")
{
    Sim2D::Options opts;
    opts.courant = 0.99;
    Sim2D sim(64, 64, opts);
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
    Sim2D::Options opts;
    opts.check_stability = false;
    opts.dt = 1.01 * courant_dt(Grid2D(64, 64, opts.dx, opts.dy), 1.0);
    Sim2D sim(64, 64, opts);
    const auto history = energy_history(sim, 20, 100);
    const double growth = history.back() / history.front();
    CAPTURE(growth);
    CHECK((!std::isfinite(growth) || growth > 1e6));
}

// Verify that a pulse travels at C_0 along an axis and along the diagonal within 1%, and
// that the axis is the slower direction, as Yee dispersion has it near the Courant limit.
TEST_CASE("Pulses travel at the speed of light", "[physics][speed]")
{
    Sim2D sim(256, 256);
    const std::vector<Cell> probes{{168, 128}, {218, 128}, {156, 156}, {191, 191}};
    const auto traces = test::support::point_source_traces(
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

// Verify that the peak of a cylindrical wave falls off as 1 / sqrt(r) in the far field,
// within 3%: dispersion spreads the pulse and lowers the peak a little with distance.
TEST_CASE("Far-field peak follows 1 / sqrt(r)", "[physics][falloff]")
{
    Sim2D sim(256, 256);
    const std::size_t distances[] = {30, 50, 70, 90, 110};
    std::vector<Cell> probes;
    for (const std::size_t r : distances) {
        probes.push_back({128 + r, 128});
    }
    const auto traces = test::support::point_source_traces(
        sim, {128, 128}, fdtd::TMz::Gaussian::from_steps(sim.dt(), 8), probes, 240);

    const double reference = test::support::find_peak(traces[0]).value * std::sqrt(30.0);
    for (std::size_t k = 1; k < probes.size(); ++k) {
        const double ratio =
            test::support::find_peak(traces[k]).value * std::sqrt(double(distances[k])) / reference;
        CAPTURE(distances[k], ratio);
        CHECK(std::abs(ratio - 1) < 0.03);
    }
}

// Verify that a closed box with a dielectric rod keeps the energy Yee conserves, within a
// bound of 1000 rounding units of FloatType over 2000 steps, under PEC and under PMC.
TEST_CASE("Closed box conserves energy", "[physics][energy]")
{
    const Boundary boundary = GENERATE(Boundary::Pec, Boundary::Pmc);
    Sim2D::Options opts;
    opts.boundary = boundary;
    Sim2D sim(128, 128, opts);
    Material2D material(sim.grid());
    material.fill_circle(80, 70, 15, {FloatType(4), FloatType(1), FloatType(0), FloatType(0)});
    sim.set_material(material);
    sim.add_source(40, 50, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim.dt(), 8));

    std::vector<double> history;
    for (std::size_t b = 0; b < 20; ++b) {
        sim.run(b == 0 ? 100 : 99);
        const auto now = sim.snapshot();
        sim.run(1);
        const auto next = sim.snapshot();
        history.push_back(test::support::field_energy(now.fields, next.fields, sim.material()));
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

// Verify that the PML settings change nothing when the PML is off: Boundary::Pml with
// thickness 0 steps as PEC, and PEC and PMC ignore the PML settings, bit for bit.
TEST_CASE("PML off reproduces the closed box bit for bit", "[physics][pml]")
{
    const Boundary boundary = GENERATE(Boundary::Pml, Boundary::Pec, Boundary::Pmc);
    Sim2D::Options opts;
    opts.boundary = boundary;
    opts.pml_thickness = boundary == Boundary::Pml ? 0 : 12;
    opts.pml_order = 2;
    opts.pml_reflection = 1e-3;
    Sim2D::Options plain;
    plain.boundary = boundary == Boundary::Pmc ? Boundary::Pmc : Boundary::Pec;

    const auto run = [](const Sim2D::Options &options) {
        Sim2D sim(48, 40, options);
        Material2D material(sim.grid());
        material.fill_circle(30, 22, 8, {FloatType(3), FloatType(1), FloatType(2), FloatType(0)});
        sim.set_material(material);
        sim.add_source(10, 12, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim.dt(), 8));
        sim.run(150);
        return sim.snapshot().fields;
    };
    CAPTURE(int(boundary));
    CHECK(same_fields(run(opts), run(plain)));
}

// Verify that a PML of 10 and 20 cells reflects less than thresholds set from measurement,
// about 10 dB above it, at normal, 45 degree and grazing incidence and from a corner.
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

// Verify that the energy of a PML box falls by five orders of magnitude between the end of
// the source and a thousand steps.
TEST_CASE("PML box loses its energy", "[physics][pml]")
{
    Sim2D::Options opts;
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 10;
    Sim2D sim(64, 64, opts);
    sim.add_source(32, 32, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim.dt(), 10));
    sim.run(60);
    const double start = conserved_energy(sim);
    sim.run(939);
    const double ratio = conserved_energy(sim) / start;
    CAPTURE(start, ratio);
    CHECK(start > 0);
    CHECK(ratio < 1e-5);
}

// Verify that the energy of a PML box keeps falling over three thousand steps, without
// the late growth an unstable PML would show.
TEST_CASE("PML box does not grow", "[physics][pml][stability]")
{
    Sim2D::Options opts;
    opts.boundary = Boundary::Pml;
    opts.pml_thickness = 10;
    Sim2D sim(64, 64, opts);
    const auto history = energy_history(sim, 30, 99);
    for (std::size_t b = 1; b < history.size(); ++b) {
        const double ratio = history[b] / history[b - 1];
        CAPTURE(b, ratio);
        REQUIRE(std::isfinite(ratio));
        CHECK(ratio < 1);
    }
}
