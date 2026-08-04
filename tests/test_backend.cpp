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

// Project header under test.
#include "fdtd/TMz/fdtd_tmz_backend.h"

// Standard library for algorithms and errors.
#include <algorithm>
#include <stdexcept>

// Verify that serial is always available and comes first.
TEST_CASE("Registry lists the serial backend first", "[unit][backend]")
{
    const auto kinds = fdtd::TMz::available_backends();
    REQUIRE_FALSE(kinds.empty());
    CHECK(kinds.front() == fdtd::TMz::BackendKind::Serial);
    CHECK(fdtd::TMz::is_backend_available(fdtd::TMz::BackendKind::Serial));
}

// Verify that only the serial backend is compiled in during M1.
TEST_CASE("Registry reports what was compiled in", "[unit][backend]")
{
    using fdtd::TMz::BackendKind;
    const fdtd::TMz::Grid2D grid(4, 4);

    CHECK(fdtd::TMz::available_backends().size() == 1);
    for (auto kind : {BackendKind::OpenMP, BackendKind::Cuda}) {
        CHECK_FALSE(fdtd::TMz::is_backend_available(kind));
        CHECK_THROWS_AS(fdtd::TMz::make_backend(kind, grid), std::runtime_error);
    }
}

// Verify the backend names.
TEST_CASE("Backend kinds have lowercase names", "[unit][backend]")
{
    using fdtd::TMz::BackendKind;
    CHECK(fdtd::TMz::backend_name(BackendKind::Serial) == "serial");
    CHECK(fdtd::TMz::backend_name(BackendKind::OpenMP) == "openmp");
    CHECK(fdtd::TMz::backend_name(BackendKind::Cuda) == "cuda");
}

// Verify that the serial backend round-trips fields and that its stub step leaves them alone.
TEST_CASE("Serial backend round-trips fields", "[unit][backend][serial]")
{
    const fdtd::TMz::Grid2D grid(7, 5);
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendKind::Serial, grid);
    REQUIRE(backend);
    CHECK(backend->name() == "serial");
    CHECK(backend->kind() == fdtd::TMz::BackendKind::Serial);
    CHECK(backend->grid().nx() == 7);

    fdtd::TMz::Fields2D input(grid);
    input.ez()[grid.index(2, 3)] = 1;
    input.hx()[grid.index(6, 0)] = 2;
    input.hy()[grid.index(0, 4)] = 3;
    backend->upload(input);
    backend->step(10);

    fdtd::TMz::Fields2D output(grid);
    backend->download(output);
    CHECK(std::equal(input.ez().begin(), input.ez().end(), output.ez().begin()));
    CHECK(std::equal(input.hx().begin(), input.hx().end(), output.hx().begin()));
    CHECK(std::equal(input.hy().begin(), input.hy().end(), output.hy().begin()));
}

// Verify that fields on another grid are rejected.
TEST_CASE("Serial backend rejects fields on another grid", "[unit][backend][serial]")
{
    auto backend = fdtd::TMz::make_backend(fdtd::TMz::BackendKind::Serial, fdtd::TMz::Grid2D(7, 5));
    fdtd::TMz::Fields2D other(fdtd::TMz::Grid2D(5, 7));
    CHECK_THROWS_AS(backend->upload(other), std::invalid_argument);
    CHECK_THROWS_AS(backend->download(other), std::invalid_argument);
}
