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
#include "fdtd/fdtd_constants.h"
#include "fdtd/fdtd_waveform.h"
#include "fdtd/test/fdtd_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using fdtd::FloatType;
using fdtd::test::Cell;
using fdtd::TMz::Boundary;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Grid2D;
using fdtd::TMz::Material2D;
using fdtd::TMz::Sim2D;

namespace
{
    // Run a centred Gaussian source and return the conserved energy after every block of
    // steps; each measurement runs one extra step for H half a step later.
    std::vector<double> energy_history(Sim2D &sim, std::size_t blocks, std::size_t blockSteps)
    {
        const std::size_t n = sim.grid().nx();
        sim.add_source(n / 2, n / 2, fdtd::TMz::Component::Ez,
                       fdtd::Gaussian::from_steps(sim.dt(), 10));
        std::vector<double> history;
        for (std::size_t b = 0; b < blocks; ++b) {
            sim.run(blockSteps);
            const auto now = sim.snapshot();
            sim.run(1);
            const auto next = sim.snapshot();
            history.push_back(fdtd::test::field_energy(now.fields, next.fields, sim.material()));
        }
        return history;
    }

    // Return the step of a trace's peak, refined between steps by a parabola through the
    // largest sample and its neighbours.
    double peak_step(const std::vector<FloatType> &trace)
    {
        const std::size_t k = fdtd::test::find_peak(trace).index;
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
    const auto traces = fdtd::test::point_source_traces(
        sim, {128, 128}, fdtd::Gaussian::from_steps(sim.dt(), 8), probes, 220);
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
    const auto traces = fdtd::test::point_source_traces(
        sim, {128, 128}, fdtd::Gaussian::from_steps(sim.dt(), 8), probes, 240);

    const double reference = fdtd::test::find_peak(traces[0]).value * std::sqrt(30.0);
    for (std::size_t k = 1; k < probes.size(); ++k) {
        const double ratio =
            fdtd::test::find_peak(traces[k]).value * std::sqrt(double(distances[k])) / reference;
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
    sim.add_source(40, 50, fdtd::TMz::Component::Ez, fdtd::Gaussian::from_steps(sim.dt(), 8));

    std::vector<double> history;
    for (std::size_t b = 0; b < 20; ++b) {
        sim.run(b == 0 ? 100 : 99);
        const auto now = sim.snapshot();
        sim.run(1);
        const auto next = sim.snapshot();
        history.push_back(fdtd::test::field_energy(now.fields, next.fields, sim.material()));
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
