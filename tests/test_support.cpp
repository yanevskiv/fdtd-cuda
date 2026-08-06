/*
 * C++ file for the physics test support tests.
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

#include "fdtd/TMz/fdtd_tmz_fields2d.h"
#include "fdtd/TMz/fdtd_tmz_material2d.h"
#include "fdtd/TMz/fdtd_tmz_sim2d.h"
#include "fdtd/fdtd_constants.h"
#include "fdtd/test/fdtd_test_support.h"

#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::test::Cell;
using fdtd::TMz::Fields2D;
using fdtd::TMz::Grid2D;
using fdtd::TMz::Material2D;
using fdtd::TMz::Sim2D;

// Verify the energy of hand-set fields in a medium with eps_r and mu_r.
TEST_CASE("field_energy sums the cell energies", "[unit][support]")
{
    const Grid2D grid(4, 3, FloatType(0.5), FloatType(0.25));
    Material2D material(grid);
    material.fill({FloatType(2), FloatType(3), FloatType(0), FloatType(0)});
    Fields2D fields(grid);
    fields.ez()[grid.index(1, 1)] = FloatType(2);
    fields.hx()[grid.index(2, 0)] = FloatType(1);
    fields.hy()[grid.index(0, 2)] = FloatType(-1);

    const double area = 0.5 * 0.25;
    const double expected = 0.5 * area * (fdtd::EPS_0 * 2 * 4 + fdtd::MU_0 * 3 * (1 + 1));
    CHECK_THAT(fdtd::test::field_energy(fields, material), WithinRel(expected, 1e-6));
}

// Verify that the two-step energy takes Ez from the first fields and multiplies H.
TEST_CASE("field_energy multiplies H over two half steps", "[unit][support]")
{
    const Grid2D grid(3, 3);
    const Material2D material(grid);
    Fields2D fields(grid);
    Fields2D next(grid);
    fields.ez()[grid.index(1, 1)] = FloatType(1);
    fields.hx()[grid.index(1, 1)] = FloatType(1);
    fields.hy()[grid.index(2, 1)] = FloatType(2);
    next.ez()[grid.index(1, 1)] = FloatType(100);
    next.hx()[grid.index(1, 1)] = FloatType(3);
    next.hy()[grid.index(2, 1)] = FloatType(-1);

    const double expected = 0.5 * (fdtd::EPS_0 * 1 + fdtd::MU_0 * (3 - 2));
    CHECK_THAT(fdtd::test::field_energy(fields, next, material), WithinRel(expected, 1e-6));
    CHECK_THROWS_AS(fdtd::test::field_energy(Fields2D(Grid2D(2, 3)), material),
                    std::invalid_argument);
}

// Verify that find_peak finds the first largest magnitude.
TEST_CASE("find_peak finds the largest magnitude", "[unit][support]")
{
    const std::vector<FloatType> values{FloatType(1), FloatType(-4), FloatType(3), FloatType(4)};
    const auto peak = fdtd::test::find_peak(values);
    CHECK(peak.index == 1);
    CHECK(peak.value == 4);
    CHECK_THROWS_AS(fdtd::test::find_peak({}), std::invalid_argument);
}

// Verify that point_source_traces returns one full trace per probe.
TEST_CASE("point_source_traces records every probe", "[unit][support]")
{
    Sim2D sim(16, 16);
    const std::vector<Cell> probes{{8, 8}, {12, 8}};
    const auto traces = fdtd::test::point_source_traces(
        sim, {8, 8}, fdtd::Gaussian::from_steps(sim.dt(), 4), probes, 20);
    REQUIRE(traces.size() == 2);
    CHECK(traces[0].size() == 20);
    CHECK(traces[1].size() == 20);
    CHECK(sim.step_count() == 20);
    CHECK(fdtd::test::find_peak(traces[0]).value > fdtd::test::find_peak(traces[1]).value);
}
