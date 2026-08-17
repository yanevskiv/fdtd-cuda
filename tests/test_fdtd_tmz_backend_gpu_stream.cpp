/*
 * C++ file for the tests of the CUDA stream.
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

// Project headers under test.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_device.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"

// Standard library for moves.
#include <utility>

// Shorthand for the type under test.
using fdtd::TMz::backends::gpu::Stream;

// Verify that a stream has a handle, waits with nothing queued, and moves its handle.
TEST_CASE("CUDA stream", "[unit][cuda]")
{
    if (fdtd::TMz::backends::gpu::device_count() == 0) {
        SKIP("no CUDA device");
    }
    Stream first;
    CHECK(first.handle() != nullptr);
    CHECK_NOTHROW(first.synchronize());

    const auto *handle = first.handle();
    Stream second(std::move(first));
    CHECK(first.handle() == nullptr);
    CHECK(second.handle() == handle);

    Stream third;
    third = std::move(second);
    CHECK(second.handle() == nullptr);
    CHECK(third.handle() == handle);
    CHECK_NOTHROW(third.synchronize());
}
