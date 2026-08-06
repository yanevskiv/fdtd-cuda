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
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "fdtd/TMz/fdtd_tmz_courant.h"
#include "fdtd/TMz/fdtd_tmz_sim2d.h"
#include "fdtd/fdtd_waveform.h"
#include "fdtd/test/fdtd_test_support.h"

#include <cmath>
#include <cstddef>
#include <vector>

using fdtd::FloatType;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Grid2D;
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
