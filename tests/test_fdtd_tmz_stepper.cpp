/*
 * C++ file for the tests of the 2D TMz stepper.
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
#include <catch2/generators/catch_generators.hpp>

// Project headers under test.
#include "fdtd/TMz/backends/fdtd_tmz_partition.h"
#include "fdtd/TMz/backends/fdtd_tmz_stepper.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_material.h"

// Standard library for algorithms, sizes and buffers.
#include <algorithm>
#include <cstddef>
#include <vector>

// Shorthand for the types under test.
using fdtd::FloatType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Coeffs;
using fdtd::TMz::Component;
using fdtd::TMz::Fields;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;
using fdtd::TMz::Partition;
using fdtd::TMz::PointInjection;
using fdtd::TMz::PointProbe;
using fdtd::TMz::Stepper;

// Helpers private to this file.
namespace
{
    // Steps of every run.
    constexpr std::size_t STEPS = 12;

    // Result of a run: the fields and the probe records.
    struct Run
    {
        /** Fields after the run. */
        Fields fields;

        /** Probe records, step-major. */
        std::vector<FloatType> records;
    };

    // Run the stepper part by part over a partition: every H phase, then every E phase.
    Run run(const Grid &grid, const Partition &partition, Boundary boundary)
    {
        const std::size_t pml = boundary == Boundary::Pml ? 4 : 0;
        Material material(grid);
        material.set_pml(pml, 3, 1e-6);
        const Coeffs coeffs(material, fdtd::TMz::courant_dt(grid, 0.9));
        const std::vector<PointInjection> sources{{Component::Ez, grid.index(9, 8)},
                                                  {Component::Hx, grid.index(12, 6)},
                                                  {Component::Hy, grid.index(6, 11)}};
        std::vector<FloatType> values(STEPS * sources.size());
        for (std::size_t k = 0; k < values.size(); ++k) {
            values[k] = FloatType(1 + k % 5);
        }
        const std::vector<PointProbe> probes{{Component::Ez, grid.index(10, 9)},
                                             {Component::Hy, grid.index(0, 0)},
                                             {Component::Ez, grid.index(10, 9)},
                                             {Component::Hx, grid.index(18, 13)}};
        Run result{Fields(grid), std::vector<FloatType>(STEPS * probes.size())};
        const Stepper stepper(result.fields, coeffs, pml, boundary, partition, sources, values,
                              probes, result.records.data());
        for (std::size_t n = 0; n < STEPS; ++n) {
            for (std::size_t p = 0; p < partition.size(); ++p) {
                stepper.h_phase(p, n);
            }
            for (std::size_t p = 0; p < partition.size(); ++p) {
                stepper.e_phase(p, n);
            }
        }
        return result;
    }

    // Check that two runs hold the same bits.
    bool same(const Run &a, const Run &b)
    {
        return std::equal(a.fields.ez().begin(), a.fields.ez().end(), b.fields.ez().begin()) &&
               std::equal(a.fields.hx().begin(), a.fields.hx().end(), b.fields.hx().begin()) &&
               std::equal(a.fields.hy().begin(), a.fields.hy().end(), b.fields.hy().begin()) &&
               std::equal(a.fields.ezx().begin(), a.fields.ezx().end(), b.fields.ezx().begin()) &&
               std::equal(a.fields.ezy().begin(), a.fields.ezy().end(), b.fields.ezy().begin()) &&
               a.records == b.records;
    }
}

// Verify that sources and probes owned by parts give the whole-grid run, bit for bit.
TEST_CASE("Stepper over parts matches the whole grid", "[unit][stepper]")
{
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Grid grid(21, 17);
    const Run whole = run(grid, Partition::whole(grid), boundary);

    CHECK(same(whole, run(grid, Partition::tiles(grid, grid.nx(), 1), boundary)));
    CHECK(same(whole, run(grid, Partition::tiles(grid, 5, 4), boundary)));
    CHECK(same(whole, run(grid, Partition::ranges(grid, 6), boundary)));
}

// Verify that the records are step-major, a repeated probe twice, and the sources land.
TEST_CASE("Stepper writes its records step-major", "[unit][stepper]")
{
    const Grid grid(21, 17);
    const Run result = run(grid, Partition::tiles(grid, 4, 4), Boundary::Pec);

    REQUIRE(result.records.size() == STEPS * 4);
    for (std::size_t n = 0; n < STEPS; ++n) {
        CHECK(result.records[n * 4] == result.records[n * 4 + 2]);
        CHECK(result.records[n * 4 + 1] == FloatType(0));
    }
    CHECK(result.records[(STEPS - 1) * 4] == result.fields.ez()[grid.index(10, 9)]);
    CHECK(result.records[(STEPS - 1) * 4 + 3] == result.fields.hx()[grid.index(18, 13)]);
    CHECK(result.fields.ez()[grid.index(9, 8)] != FloatType(0));
}
