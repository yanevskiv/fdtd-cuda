/*
 * C++ file for the 2D TMz backend registry tests.
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
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers under test.
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"

// Standard library for algorithms, sizes and errors.
#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

// Shorthand for the types under test.
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Coeffs;
using fdtd::TMz::Component;
using fdtd::TMz::Fields;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;
using fdtd::TMz::PointInjection;
using fdtd::TMz::PointProbe;

// Create a serial backend in vacuum at a Courant number, with a unit Ez impulse at (ci, cj).
static auto make_impulse(const Grid &grid, double courant, std::size_t ci, std::size_t cj, Boundary boundary)
{
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, grid);
    backend->upload(Coeffs(Material(grid), fdtd::TMz::courant_dt(grid, courant)));
    backend->set_boundary(boundary);
    Fields fields(grid);
    fields.ez()[grid.index(ci, cj)] = 1;
    backend->upload(fields);
    return backend;
}

// Download a backend's fields.
static Fields download(const fdtd::TMz::Backend &backend)
{
    Fields fields(backend.grid());
    backend.download(fields);
    return fields;
}

// Verify that serial is always available and comes first.
TEST_CASE("Registry lists the serial backend first", "[unit][backend]")
{
    const auto types = fdtd::TMz::available_backends();
    REQUIRE_FALSE(types.empty());
    CHECK(types.front() == fdtd::TMz::BackendType::Serial);
    CHECK(fdtd::TMz::is_backend_available(fdtd::TMz::BackendType::Serial));
}

// Verify that the registry lists OpenMP when compiled in, and CUDA only when compiled in.
TEST_CASE("Registry reports what was compiled in", "[unit][backend]")
{
    using fdtd::TMz::BackendType;
    const fdtd::TMz::Grid grid(4, 4);

    const bool cuda = fdtd::TMz::is_backend_available(BackendType::Cuda);
    CHECK(fdtd::TMz::available_backends().size() == 1 + (FDTD_WITH_OPENMP ? 1U : 0U) + (cuda ? 1U : 0U));
    CHECK(fdtd::TMz::is_backend_available(BackendType::OpenMP) == bool(FDTD_WITH_OPENMP));
    if (!FDTD_WITH_OPENMP) {
        CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::OpenMP, grid), std::runtime_error);
    }
    if (!FDTD_WITH_CUDA) {
        CHECK_FALSE(cuda);
        CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::Cuda, grid), std::runtime_error);
    }
}

// Verify that the serial backend ignores the thread count and reads no variant but the default.
TEST_CASE("Serial backend ignores the thread count", "[unit][backend][serial]")
{
    const Grid grid(9, 7);
    fdtd::TMz::Simulation::Options options;
    options.threads = 3;
    fdtd::TMz::Simulation::Options level = options;
    level.variant = "v1";
    CHECK_THROWS_AS(fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, grid, level), std::invalid_argument);
    auto plain = make_impulse(grid, 0.9, 4, 3, Boundary::Pmc);
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, grid, options);
    backend->upload(Coeffs(Material(grid), fdtd::TMz::courant_dt(grid, 0.9)));
    backend->set_boundary(Boundary::Pmc);
    backend->upload(download(*plain));

    plain->step(5);
    backend->step(5);

    const Fields a = download(*plain);
    const Fields b = download(*backend);
    CHECK(std::equal(a.ez().begin(), a.ez().end(), b.ez().begin()));
    CHECK(std::equal(a.hx().begin(), a.hx().end(), b.hx().begin()));
    CHECK(std::equal(a.hy().begin(), a.hy().end(), b.hy().begin()));
}

// Verify the traffic model of every available backend against a count by hand on an 8 by 6 grid with a PML 2 cells thick: 48 Ez reads, 40
// Hx and 42 Hy updates at 4 words and 5 flops, 24 edge cells, and 24 cells off the edge, 8 of them interior and 16 in the PML when it is
// active.
TEST_CASE("Backends model the traffic of a step", "[unit][backend]")
{
    const double word = sizeof(FloatType);
    struct Case
    {
        Boundary boundary;
        double words;
        double flops;
    };
    const std::array cases{
        Case{Boundary::Pml, 48 + 4 * 82 + 8 * 6 + 16 * 11 + 24 * 3, 5 * 82 + 8 * 8 + 16 * 11},
        Case{Boundary::Pec, 48 + 4 * 82 + 24 * 6 + 24 * 3, 5 * 82 + 24 * 8},
        Case{Boundary::Pmc, 48 + 4 * 82 + 24 * 6 + 24 * 6, 5 * 82 + 24 * 8 + 24 * 8},
    };
    for (const auto type : fdtd::TMz::available_backends()) {
        for (const Case &c : cases) {
            INFO(fdtd::TMz::backend_name(type) << ", boundary " << static_cast<int>(c.boundary));
            fdtd::TMz::Simulation::Options options;
            options.backend = type;
            if (type == fdtd::TMz::BackendType::OpenMP) {
                options.variant = "v2";
            } else if (type == fdtd::TMz::BackendType::Cuda) {
                options.variant = "fused";
            }
            options.boundary = c.boundary;
            options.pml_thickness = 2;
            const fdtd::TMz::Simulation sim(8, 6, options);
            const auto traffic = sim.step_traffic();
            CHECK(traffic.bytes == c.words * word);
            CHECK(traffic.flops == c.flops);
            const auto model = fdtd::TMz::two_pass_traffic(Grid(8, 6), c.boundary == Boundary::Pml ? 2 : 0, c.boundary);
            CHECK(model.bytes == traffic.bytes);
        }
    }
}

// Verify that a step with snapshots hands over each one in order and checks the steps.
TEST_CASE("Backends hand over snapshots and check their steps", "[unit][backend]")
{
    const Grid grid(10, 8);
    const Coeffs coeffs(Material(grid), fdtd::TMz::courant_dt(grid, 0.9));
    const std::vector<PointInjection> points{{Component::Ez, grid.index(5, 4)}};
    const std::vector<FloatType> values(6, FloatType(1));
    for (const auto type : fdtd::TMz::available_backends()) {
        INFO(fdtd::TMz::backend_name(type));
        const auto backend = fdtd::TMz::make_backend(type, grid);
        backend->upload(coeffs);
        std::vector<std::size_t> taken;
        std::vector<FloatType> centre;
        const std::vector<std::size_t> snapshots{1, 2, 6};
        backend->step(6, points, values, snapshots, [&](std::size_t steps, const Fields &fields) {
            taken.push_back(steps);
            centre.push_back(fields.ez()[grid.index(5, 4)]);
        });
        CHECK(taken == snapshots);
        CHECK(centre[0] == FloatType(1));
        for (const std::vector<std::size_t> &bad :
             {std::vector<std::size_t>{0}, std::vector<std::size_t>{7}, std::vector<std::size_t>{2, 2}, std::vector<std::size_t>{3, 1}}) {
            CHECK_THROWS_AS(backend->step(6, points, values, bad, [](std::size_t, const Fields &) {}), std::invalid_argument);
        }
    }
}

// Verify the backend names.
TEST_CASE("Backend types have lowercase names", "[unit][backend]")
{
    using fdtd::TMz::BackendType;
    CHECK(fdtd::TMz::backend_name(BackendType::Serial) == "serial");
    CHECK(fdtd::TMz::backend_name(BackendType::OpenMP) == "openmp");
    CHECK(fdtd::TMz::backend_name(BackendType::Cuda) == "cuda");
}

// Verify that the serial backend round-trips fields, the split parts of Ez included.
TEST_CASE("Serial backend round-trips fields", "[unit][backend][serial]")
{
    const fdtd::TMz::Grid grid(7, 5);
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, grid);
    REQUIRE(backend);
    CHECK(backend->name() == "serial");
    CHECK(backend->type() == fdtd::TMz::BackendType::Serial);
    CHECK(backend->grid().nx() == 7);

    fdtd::TMz::Fields input(grid);
    input.ez()[grid.index(2, 3)] = 1;
    input.hx()[grid.index(6, 0)] = 2;
    input.hy()[grid.index(0, 4)] = 3;
    input.ezx()[grid.index(1, 1)] = 4;
    input.ezy()[grid.index(5, 2)] = 5;
    backend->upload(input);

    fdtd::TMz::Fields output(grid);
    backend->download(output);
    CHECK(std::equal(input.ez().begin(), input.ez().end(), output.ez().begin()));
    CHECK(std::equal(input.hx().begin(), input.hx().end(), output.hx().begin()));
    CHECK(std::equal(input.hy().begin(), input.hy().end(), output.hy().begin()));
    CHECK(std::equal(input.ezx().begin(), input.ezx().end(), output.ezx().begin()));
    CHECK(std::equal(input.ezy().begin(), input.ezy().end(), output.ezy().begin()));
}

// Verify that fields on another grid are rejected.
TEST_CASE("Serial backend rejects fields on another grid", "[unit][backend][serial]")
{
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, fdtd::TMz::Grid(7, 5));
    fdtd::TMz::Fields other(fdtd::TMz::Grid(5, 7));
    CHECK_THROWS_AS(backend->upload(other), std::invalid_argument);
    CHECK_THROWS_AS(backend->download(other), std::invalid_argument);
}

// Verify the coefficient upload checks and the boundary default.
TEST_CASE("Serial backend takes coefficients and a boundary", "[unit][backend][serial]")
{
    const Grid grid(7, 5);
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, grid);

    CHECK(backend->boundary() == Boundary::Pec);
    CHECK_THROWS_AS(backend->step(1), std::logic_error);

    const Grid other(5, 7);
    CHECK_THROWS_AS(backend->upload(Coeffs(Material(other), 1e-12)), std::invalid_argument);

    backend->upload(Coeffs(Material(grid), fdtd::TMz::courant_dt(grid, 0.9)));
    backend->set_boundary(Boundary::Pmc);
    CHECK(backend->boundary() == Boundary::Pmc);
    CHECK_NOTHROW(backend->step(1));
}

// Verify one step from a unit impulse at S = 1/2: the centre falls to 1/2, neighbours rise to 1/8.
TEST_CASE("Serial backend steps an impulse", "[unit][backend][serial]")
{
    const Grid grid(7, 7, 1e-3F, 1e-3F);
    auto backend = make_impulse(grid, 0.5, 3, 3, Boundary::Pec);
    backend->step(1);
    const Fields f = download(*backend);

    CHECK_THAT(double(f.ez()[grid.index(3, 3)]), WithinRel(0.5, 1e-5));
    CHECK_THAT(double(f.ez()[grid.index(2, 3)]), WithinRel(0.125, 1e-5));
    CHECK_THAT(double(f.ez()[grid.index(4, 3)]), WithinRel(0.125, 1e-5));
    CHECK_THAT(double(f.ez()[grid.index(3, 2)]), WithinRel(0.125, 1e-5));
    CHECK_THAT(double(f.ez()[grid.index(3, 4)]), WithinRel(0.125, 1e-5));
}

// Verify that under PEC the edge Ez is zero after every step, even when uploaded non-zero.
TEST_CASE("Serial backend holds the edge Ez at zero under PEC", "[unit][backend][serial]")
{
    const Grid grid(6, 5);
    auto backend = make_impulse(grid, 0.9, 0, 2, Boundary::Pec);
    for (int n = 0; n < 15; ++n) {
        backend->step(1);
        const Fields f = download(*backend);
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            CHECK(f.ez()[grid.index(i, 0)] == fdtd::FloatType(0));
            CHECK(f.ez()[grid.index(i, grid.ny() - 1)] == fdtd::FloatType(0));
        }
        for (std::size_t j = 0; j < grid.ny(); ++j) {
            CHECK(f.ez()[grid.index(0, j)] == fdtd::FloatType(0));
            CHECK(f.ez()[grid.index(grid.nx() - 1, j)] == fdtd::FloatType(0));
        }
    }
}

// Helpers private to this file.
namespace
{
    // Symmetries an impulse keeps.
    enum class Symmetry {
        // The mirrors along x and y.
        Mirrors,

        // The mirrors and the diagonal.
        Diagonal
    };
}

// Verify that a centred impulse keeps its mirror symmetries, and the diagonal one on square cells.
TEST_CASE("Serial backend keeps a centred impulse symmetric", "[unit][backend][serial]")
{
    const Boundary boundary = GENERATE(Boundary::Pec, Boundary::Pmc);
    const auto symmetric = [](const Grid &grid, Symmetry symmetry, Boundary b) {
        const std::size_t mx = grid.nx() - 1;
        const std::size_t my = grid.ny() - 1;
        auto backend = make_impulse(grid, 0.8, mx / 2, my / 2, b);
        backend->step(40);
        const Fields f = download(*backend);
        const auto ez = [&](std::size_t i, std::size_t j) {
            return double(f.ez()[grid.index(i, j)]);
        };
        for (std::size_t j = 0; j <= my; ++j) {
            for (std::size_t i = 0; i <= mx; ++i) {
                const double v = ez(i, j);
                CHECK_THAT(ez(mx - i, j), WithinAbs(v, 1e-6));
                CHECK_THAT(ez(i, my - j), WithinAbs(v, 1e-6));
                CHECK_THAT(ez(mx - i, my - j), WithinAbs(v, 1e-6));
                if (symmetry == Symmetry::Diagonal) {
                    CHECK_THAT(ez(j, i), WithinAbs(v, 1e-6));
                }
            }
        }
    };

    symmetric(Grid(13, 9, 1e-3F, 2e-3F), Symmetry::Mirrors, boundary);
    symmetric(Grid(11, 11, 1e-3F, 1e-3F), Symmetry::Diagonal, boundary);
}

// Verify that step(a) then step(b) matches step(a + b) bit for bit.
TEST_CASE("Serial backend steps compose", "[unit][backend][serial]")
{
    const Grid grid(10, 8);
    auto split = make_impulse(grid, 0.9, 3, 4, Boundary::Pmc);
    auto whole = make_impulse(grid, 0.9, 3, 4, Boundary::Pmc);
    split->step(7);
    split->step(5);
    whole->step(12);
    const Fields a = download(*split);
    const Fields b = download(*whole);

    CHECK(std::equal(a.ez().begin(), a.ez().end(), b.ez().begin()));
    CHECK(std::equal(a.hx().begin(), a.hx().end(), b.hx().begin()));
    CHECK(std::equal(a.hy().begin(), a.hy().end(), b.hy().begin()));
}

// Verify that each source value lands on its cell, the H sources before the E update.
TEST_CASE("Serial backend adds point sources", "[unit][backend][serial]")
{
    const Grid grid(9, 7);
    auto backend = make_impulse(grid, 0.5, 4, 3, Boundary::Pec);
    backend->upload(Fields(grid));

    const std::array points{PointInjection{Component::Ez, grid.index(4, 3)}, PointInjection{Component::Hx, grid.index(1, 1)},
                            PointInjection{Component::Hy, grid.index(6, 5)}};
    const std::array values{FloatType(2), FloatType(3), FloatType(5)};
    backend->step(1, points, values);
    const Fields f = download(*backend);

    CHECK(f.ez()[grid.index(4, 3)] == FloatType(2));
    CHECK(f.hx()[grid.index(1, 1)] == FloatType(3));
    CHECK(f.hy()[grid.index(6, 5)] == FloatType(5));
    CHECK(f.ez()[grid.index(1, 1)] != FloatType(0));
    CHECK(f.ez()[grid.index(6, 5)] != FloatType(0));
}

// Verify that source values are read step-major.
TEST_CASE("Serial backend reads source values per step", "[unit][backend][serial]")
{
    const Grid grid(9, 9);
    auto backend = make_impulse(grid, 0.5, 4, 4, Boundary::Pec);
    backend->upload(Fields(grid));

    const std::array points{PointInjection{Component::Ez, grid.index(4, 4)}};
    const std::array values{FloatType(0), FloatType(1)};
    backend->step(2, points, values);
    const Fields f = download(*backend);

    CHECK(f.ez()[grid.index(4, 4)] == FloatType(1));
    CHECK(f.ez()[grid.index(3, 4)] == FloatType(0));
}

// Verify that bad source arguments throw before any step runs.
TEST_CASE("Serial backend rejects bad sources", "[unit][backend][serial]")
{
    const Grid grid(9, 7);
    auto backend = make_impulse(grid, 0.5, 4, 3, Boundary::Pec);
    const PointInjection centre{Component::Ez, grid.index(4, 3)};
    const std::array one{centre};
    const std::array value{FloatType(1)};
    const std::array twoValues{FloatType(1), FloatType(1)};

    CHECK_THROWS_AS(backend->step(1, one, twoValues), std::invalid_argument);
    CHECK_THROWS_AS(backend->step(2, one, value), std::invalid_argument);

    const std::array outside{PointInjection{Component::Ez, grid.size()}};
    CHECK_THROWS_AS(backend->step(1, outside, value), std::invalid_argument);
    if (grid.pitch() > grid.nx()) {
        const std::array padding{PointInjection{Component::Ez, grid.nx()}};
        CHECK_THROWS_AS(backend->step(1, padding, value), std::invalid_argument);
    }

    const std::array twice{centre, centre};
    CHECK_THROWS_AS(backend->step(1, twice, twoValues), std::invalid_argument);
    const std::array distinct{centre, PointInjection{Component::Hx, centre.index}};
    CHECK_NOTHROW(backend->step(1, distinct, twoValues));
    CHECK_NOTHROW(backend->step(0, distinct, std::span<const FloatType>{}));
}

// Verify that each step appends every probe's end-of-step value and that resetting clears them.
TEST_CASE("Serial backend records probes", "[unit][backend][serial]")
{
    const Grid grid(9, 9);
    auto backend = make_impulse(grid, 0.5, 4, 4, Boundary::Pec);
    const std::array probes{PointProbe{Component::Ez, grid.index(4, 4)}, PointProbe{Component::Hx, grid.index(4, 3)},
                            PointProbe{Component::Hy, grid.index(3, 4)}, PointProbe{Component::Ez, grid.index(4, 4)}};
    backend->set_probes(probes);
    CHECK(backend->probe_count() == 4);
    CHECK(backend->probe_steps() == 0);

    std::vector<FloatType> expected;
    for (int n = 0; n < 3; ++n) {
        backend->step(1);
        const Fields f = download(*backend);
        expected.push_back(f.ez()[grid.index(4, 4)]);
        expected.push_back(f.hx()[grid.index(4, 3)]);
        expected.push_back(f.hy()[grid.index(3, 4)]);
        expected.push_back(f.ez()[grid.index(4, 4)]);
    }
    CHECK(backend->probe_steps() == 3);
    CHECK(backend->fetch_probes() == expected);
    CHECK(expected[1] != FloatType(0));
    CHECK(expected[2] != FloatType(0));

    backend->set_probes(std::span(probes).first(1));
    CHECK(backend->probe_count() == 1);
    CHECK(backend->probe_steps() == 0);
    CHECK(backend->fetch_probes().empty());
}

// Verify that probes outside the grid or in the padding throw and keep the old probes.
TEST_CASE("Serial backend rejects probes outside the grid", "[unit][backend][serial]")
{
    const Grid grid(9, 7);
    auto backend = make_impulse(grid, 0.5, 4, 3, Boundary::Pec);
    const std::array inside{PointProbe{Component::Ez, grid.index(4, 3)}};
    backend->set_probes(inside);

    const std::array outside{PointProbe{Component::Hx, grid.size()}};
    CHECK_THROWS_AS(backend->set_probes(outside), std::invalid_argument);
    if (grid.pitch() > grid.nx()) {
        const std::array padding{PointProbe{Component::Ez, grid.nx()}};
        CHECK_THROWS_AS(backend->set_probes(padding), std::invalid_argument);
    }
    CHECK(backend->probe_count() == 1);
}

// Verify that only Boundary::Pml runs the split update, Ez being the sum of its parts there.
TEST_CASE("Serial backend splits Ez in the PML only under Boundary::Pml", "[unit][backend][serial][pml]")
{
    const std::size_t pml = 4;
    const Grid grid(16, 16);
    Material material(grid);
    material.set_pml(pml, 3, 1e-6);
    const Boundary boundary = GENERATE(Boundary::Pml, Boundary::Pec);
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, grid);
    backend->upload(Coeffs(material, fdtd::TMz::courant_dt(grid, 0.9)));
    backend->set_boundary(boundary);
    Fields fields(grid);
    fields.ez()[grid.index(8, 8)] = 1;
    backend->upload(fields);
    backend->step(20);
    const Fields f = download(*backend);

    const bool split = boundary == Boundary::Pml;
    bool anyPart = false;
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            const std::size_t k = grid.index(i, j);
            const bool edge = i == 0 || j == 0 || i + 1 == grid.nx() || j + 1 == grid.ny();
            const bool slab = i < pml || j < pml || i + pml >= grid.nx() || j + pml >= grid.ny();
            if (edge) {
                CHECK(f.ez()[k] == FloatType(0));
                CHECK(f.ezx()[k] == FloatType(0));
                CHECK(f.ezy()[k] == FloatType(0));
            } else if (slab && split) {
                CHECK(f.ez()[k] == f.ezx()[k] + f.ezy()[k]);
            } else {
                CHECK(f.ezx()[k] == FloatType(0));
                CHECK(f.ezy()[k] == FloatType(0));
            }
            anyPart = anyPart || f.ezx()[k] != FloatType(0);
        }
    }
    CHECK(anyPart == split);
}

// Verify which backend reads which name, whether compiled in or not, and the name each writes.
TEST_CASE("Variant names belong to their backend", "[unit][backend]")
{
    using fdtd::TMz::BackendType;
    using fdtd::TMz::reads_variant;
    using fdtd::TMz::variant_name;
    for (const BackendType backend : {BackendType::Serial, BackendType::OpenMP, BackendType::Cuda}) {
        CHECK(reads_variant(backend, "default"));
        CHECK_FALSE(reads_variant(backend, "fast"));
    }
    CHECK(reads_variant(BackendType::OpenMP, "v2+close"));
    CHECK_FALSE(reads_variant(BackendType::Cuda, "v2+close"));
    CHECK_FALSE(reads_variant(BackendType::Serial, "v1"));
    CHECK(reads_variant(BackendType::Cuda, "tiled:16x16"));
    CHECK_FALSE(reads_variant(BackendType::OpenMP, "tiled:16x16"));
    CHECK_FALSE(reads_variant(BackendType::Serial, "fused"));

    CHECK(variant_name(BackendType::Serial, "default") == "default");
    CHECK(variant_name(BackendType::OpenMP, "v1") == "v1");
    CHECK(variant_name(BackendType::OpenMP, "default") == "v4");
    CHECK(variant_name(BackendType::Cuda, "tiled:8x4") == "tiled:8x4");
    CHECK_THROWS_AS(variant_name(BackendType::Serial, "v1"), std::invalid_argument);
    CHECK_THROWS_AS(variant_name(BackendType::OpenMP, "fused"), std::invalid_argument);
}
