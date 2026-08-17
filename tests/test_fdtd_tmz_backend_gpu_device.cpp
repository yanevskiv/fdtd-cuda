/*
 * C++ file for the tests of the queries of the CUDA device.
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
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_device.h"

// Standard library for sizes.
#include <cstddef>

// Verify that the device count is a plain answer, the same each time it is asked.
TEST_CASE("CUDA device count", "[unit][cuda]")
{
    const std::size_t count = fdtd::TMz::backends::gpu::device_count();
    if (count == 0) {
        SKIP("no CUDA device");
    }
    CHECK(fdtd::TMz::backends::gpu::device_count() == count);
}
