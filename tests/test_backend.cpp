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
#include "fdtd/TMz/fdtd_tmz_courant.h"

// Standard library for algorithms, sizes and errors.
#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

// Shorthand for the types under test.
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Coeffs2D;
using fdtd::TMz::Component;
using fdtd::TMz::Fields2D;
using fdtd::TMz::Grid2D;
using fdtd::TMz::Material2D;
using fdtd::TMz::PointInjection;

// Create a serial backend in vacuum at a Courant number, with a unit Ez impulse at (ci, cj).
static auto make_impulse(const Grid2D &grid, double courant, std::size_t ci, std::size_t cj,
                         Boundary boundary)
{
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, grid);
    backend->upload(Coeffs2D(Material2D(grid), fdtd::TMz::courant_dt(grid, courant)));
    backend->set_boundary(boundary);
    Fields2D fields(grid);
    fields.ez()[grid.index(ci, cj)] = 1;
    backend->upload(fields);
    return backend;
}

// Download a backend's fields.
static Fields2D download(const fdtd::TMz::Backend &backend)
{
    Fields2D fields(backend.grid());
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

// Verify that only the serial backend is compiled in during M1.
TEST_CASE("Registry reports what was compiled in", "[unit][backend]")
{
    using fdtd::TMz::BackendType;
    const fdtd::TMz::Grid2D grid(4, 4);

    CHECK(fdtd::TMz::available_backends().size() == 1);
    for (auto type : {BackendType::OpenMP, BackendType::Cuda}) {
        CHECK_FALSE(fdtd::TMz::is_backend_available(type));
        CHECK_THROWS_AS(fdtd::TMz::make_backend(type, grid), std::runtime_error);
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

// Verify that the serial backend round-trips fields.
TEST_CASE("Serial backend round-trips fields", "[unit][backend][serial]")
{
    const fdtd::TMz::Grid2D grid(7, 5);
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, grid);
    REQUIRE(backend);
    CHECK(backend->name() == "serial");
    CHECK(backend->type() == fdtd::TMz::BackendType::Serial);
    CHECK(backend->grid().nx() == 7);

    fdtd::TMz::Fields2D input(grid);
    input.ez()[grid.index(2, 3)] = 1;
    input.hx()[grid.index(6, 0)] = 2;
    input.hy()[grid.index(0, 4)] = 3;
    backend->upload(input);

    fdtd::TMz::Fields2D output(grid);
    backend->download(output);
    CHECK(std::equal(input.ez().begin(), input.ez().end(), output.ez().begin()));
    CHECK(std::equal(input.hx().begin(), input.hx().end(), output.hx().begin()));
    CHECK(std::equal(input.hy().begin(), input.hy().end(), output.hy().begin()));
}

// Verify that fields on another grid are rejected.
TEST_CASE("Serial backend rejects fields on another grid", "[unit][backend][serial]")
{
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, fdtd::TMz::Grid2D(7, 5));
    fdtd::TMz::Fields2D other(fdtd::TMz::Grid2D(5, 7));
    CHECK_THROWS_AS(backend->upload(other), std::invalid_argument);
    CHECK_THROWS_AS(backend->download(other), std::invalid_argument);
}

// Verify the coefficient upload checks and the boundary default.
TEST_CASE("Serial backend takes coefficients and a boundary", "[unit][backend][serial]")
{
    const Grid2D grid(7, 5);
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendType::Serial, grid);

    CHECK(backend->boundary() == Boundary::Pec);
    CHECK_THROWS_AS(backend->step(1), std::logic_error);

    const Grid2D other(5, 7);
    CHECK_THROWS_AS(backend->upload(Coeffs2D(Material2D(other), 1e-12)), std::invalid_argument);

    backend->upload(Coeffs2D(Material2D(grid), fdtd::TMz::courant_dt(grid, 0.9)));
    backend->set_boundary(Boundary::Pmc);
    CHECK(backend->boundary() == Boundary::Pmc);
    CHECK_NOTHROW(backend->step(1));
}

// Verify one step from a unit impulse in vacuum on square cells with S = 1/2: the centre
// falls to 1/2 and each neighbour rises to 1/8, as c^2 dt^2 / dx^2 = S^2 / 2.
TEST_CASE("Serial backend steps an impulse", "[unit][backend][serial]")
{
    const Grid2D grid(7, 7, 1e-3F, 1e-3F);
    auto backend = make_impulse(grid, 0.5, 3, 3, Boundary::Pec);
    backend->step(1);
    const Fields2D f = download(*backend);

    CHECK_THAT(double(f.ez()[grid.index(3, 3)]), WithinRel(0.5, 1e-5));
    CHECK_THAT(double(f.ez()[grid.index(2, 3)]), WithinRel(0.125, 1e-5));
    CHECK_THAT(double(f.ez()[grid.index(4, 3)]), WithinRel(0.125, 1e-5));
    CHECK_THAT(double(f.ez()[grid.index(3, 2)]), WithinRel(0.125, 1e-5));
    CHECK_THAT(double(f.ez()[grid.index(3, 4)]), WithinRel(0.125, 1e-5));
}

// Verify that under PEC the edge Ez is zero after every step, even when uploaded non-zero.
TEST_CASE("Serial backend holds the edge Ez at zero under PEC", "[unit][backend][serial]")
{
    const Grid2D grid(6, 5);
    auto backend = make_impulse(grid, 0.9, 0, 2, Boundary::Pec);
    for (int n = 0; n < 15; ++n) {
        backend->step(1);
        const Fields2D f = download(*backend);
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

// Verify that a centred impulse stays symmetric about x, about y and about the centre on
// rectangular cells, and about the diagonal on square ones, under both boundaries.
TEST_CASE("Serial backend keeps a centred impulse symmetric", "[unit][backend][serial]")
{
    const Boundary boundary = GENERATE(Boundary::Pec, Boundary::Pmc);
    const auto symmetric = [](const Grid2D &grid, bool diagonal, Boundary b) {
        const std::size_t mx = grid.nx() - 1;
        const std::size_t my = grid.ny() - 1;
        auto backend = make_impulse(grid, 0.8, mx / 2, my / 2, b);
        backend->step(40);
        const Fields2D f = download(*backend);
        const auto ez = [&](std::size_t i, std::size_t j) {
            return double(f.ez()[grid.index(i, j)]);
        };
        for (std::size_t j = 0; j <= my; ++j) {
            for (std::size_t i = 0; i <= mx; ++i) {
                const double v = ez(i, j);
                CHECK_THAT(ez(mx - i, j), WithinAbs(v, 1e-6));
                CHECK_THAT(ez(i, my - j), WithinAbs(v, 1e-6));
                CHECK_THAT(ez(mx - i, my - j), WithinAbs(v, 1e-6));
                if (diagonal) {
                    CHECK_THAT(ez(j, i), WithinAbs(v, 1e-6));
                }
            }
        }
    };

    symmetric(Grid2D(13, 9, 1e-3F, 2e-3F), false, boundary);
    symmetric(Grid2D(11, 11, 1e-3F, 1e-3F), true, boundary);
}

// Verify that step(a) then step(b) matches step(a + b) bit for bit.
TEST_CASE("Serial backend steps compose", "[unit][backend][serial]")
{
    const Grid2D grid(10, 8);
    auto split = make_impulse(grid, 0.9, 3, 4, Boundary::Pmc);
    auto whole = make_impulse(grid, 0.9, 3, 4, Boundary::Pmc);
    split->step(7);
    split->step(5);
    whole->step(12);
    const Fields2D a = download(*split);
    const Fields2D b = download(*whole);

    CHECK(std::equal(a.ez().begin(), a.ez().end(), b.ez().begin()));
    CHECK(std::equal(a.hx().begin(), a.hx().end(), b.hx().begin()));
    CHECK(std::equal(a.hy().begin(), a.hy().end(), b.hy().begin()));
}

// Verify that each source value lands on its cell: Ez after the edge update, H after the
// H update, so a step from zero fields leaves exactly the injected values there; the H
// sources land before the E update, so the Ez next to them moves.
TEST_CASE("Serial backend adds point sources", "[unit][backend][serial]")
{
    const Grid2D grid(9, 7);
    auto backend = make_impulse(grid, 0.5, 4, 3, Boundary::Pec);
    backend->upload(Fields2D(grid));

    const std::array points{PointInjection{Component::Ez, grid.index(4, 3)},
                            PointInjection{Component::Hx, grid.index(1, 1)},
                            PointInjection{Component::Hy, grid.index(6, 5)}};
    const std::array values{FloatType(2), FloatType(3), FloatType(5)};
    backend->step(1, points, values);
    const Fields2D f = download(*backend);

    CHECK(f.ez()[grid.index(4, 3)] == FloatType(2));
    CHECK(f.hx()[grid.index(1, 1)] == FloatType(3));
    CHECK(f.hy()[grid.index(6, 5)] == FloatType(5));
    CHECK(f.ez()[grid.index(1, 1)] != FloatType(0));
    CHECK(f.ez()[grid.index(6, 5)] != FloatType(0));
}

// Verify that values are read step-major: a zero in step 0 and a one in step 1 leave only
// the one, since nothing has moved yet when it lands.
TEST_CASE("Serial backend reads source values per step", "[unit][backend][serial]")
{
    const Grid2D grid(9, 9);
    auto backend = make_impulse(grid, 0.5, 4, 4, Boundary::Pec);
    backend->upload(Fields2D(grid));

    const std::array points{PointInjection{Component::Ez, grid.index(4, 4)}};
    const std::array values{FloatType(0), FloatType(1)};
    backend->step(2, points, values);
    const Fields2D f = download(*backend);

    CHECK(f.ez()[grid.index(4, 4)] == FloatType(1));
    CHECK(f.ez()[grid.index(3, 4)] == FloatType(0));
}

// Verify that bad source arguments throw before any step runs.
TEST_CASE("Serial backend rejects bad sources", "[unit][backend][serial]")
{
    const Grid2D grid(9, 7);
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
