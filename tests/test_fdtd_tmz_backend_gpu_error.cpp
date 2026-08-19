/*
 * C++ file for the tests of the error a failed CUDA call throws.
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
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_error.h"

// Standard library for errors and strings.
#include <stdexcept>
#include <string>

// Verify that the error keeps its code and message and is a runtime error.
TEST_CASE("CudaError keeps its code and message", "[unit][cuda]")
{
    const fdtd::TMz::backends::gpu::CudaError error(2, "cudaMalloc: out of memory");
    CHECK(error.code() == 2);
    CHECK(std::string(error.what()) == "cudaMalloc: out of memory");

    const std::runtime_error &base = error;
    CHECK(std::string(base.what()) == "cudaMalloc: out of memory");
}
