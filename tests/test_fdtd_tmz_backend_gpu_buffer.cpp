/*
 * C++ file for the tests of the buffers in CUDA device memory.
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
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_buffer.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_device.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_error.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"
#include "fdtd/fdtd_float.h"

// Standard library for sizes, errors, moves and buffers.
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// Shorthand for the types under test.
using fdtd::FloatType;
using fdtd::TMz::backends::gpu::DeviceBuffer;
using fdtd::TMz::backends::gpu::Stream;

// Helpers private to this file.
namespace
{
    // Skip the test when no CUDA device is present.
    void require_device()
    {
        if (fdtd::TMz::backends::gpu::device_count() == 0) {
            SKIP("no CUDA device");
        }
    }
}

// Verify that the pitch pads a row to a multiple of 128 bytes and never shrinks it.
TEST_CASE("Device pitch pads rows to 128 bytes", "[unit][cuda]")
{
    CHECK(fdtd::TMz::backends::gpu::device_pitch(0, 4) == 0);
    CHECK(fdtd::TMz::backends::gpu::device_pitch(1, 4) == 32);
    CHECK(fdtd::TMz::backends::gpu::device_pitch(32, 4) == 32);
    CHECK(fdtd::TMz::backends::gpu::device_pitch(33, 4) == 64);
    CHECK(fdtd::TMz::backends::gpu::device_pitch(33, 8) == 48);
    CHECK(fdtd::TMz::backends::gpu::device_pitch(100, 8) == 112);
}

// Verify that rows survive a round trip through a padded device buffer, the padding unseen.
TEST_CASE("Device buffer round trip with padded rows", "[unit][cuda]")
{
    require_device();
    const auto nx = GENERATE(std::size_t(1), std::size_t(31), std::size_t(67));
    const auto ny = GENERATE(std::size_t(1), std::size_t(5), std::size_t(45));
    const std::size_t pitch = fdtd::TMz::backends::gpu::device_pitch(nx, sizeof(FloatType));
    std::vector<FloatType> host(nx * ny);
    for (std::size_t k = 0; k < host.size(); ++k) {
        host[k] = FloatType(k) / FloatType(7);
    }

    const Stream stream;
    DeviceBuffer<FloatType> device(pitch * ny);
    device.zero(stream);
    fdtd::TMz::backends::gpu::upload_rows(device, pitch, std::span<const FloatType>(host), nx, nx,
                                          ny, stream);
    std::vector<FloatType> back(nx * ny);
    fdtd::TMz::backends::gpu::download_rows(std::span<FloatType>(back), nx, device, pitch, nx, ny,
                                            stream);
    CHECK(back == host);

    std::vector<FloatType> padded(pitch * ny, FloatType(-1));
    fdtd::TMz::backends::gpu::download(std::span<FloatType>(padded), device, stream);
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = nx; i < pitch; ++i) {
            CHECK(padded[j * pitch + i] == FloatType(0));
        }
    }
}

// Verify a flat upload and download, and that zero() clears the buffer.
TEST_CASE("Device buffer flat copies and zero", "[unit][cuda]")
{
    require_device();
    const Stream stream;
    const std::vector<FloatType> host{FloatType(1), FloatType(2), FloatType(3)};
    DeviceBuffer<FloatType> device(host.size());
    fdtd::TMz::backends::gpu::upload(device, std::span<const FloatType>(host), stream);
    std::vector<FloatType> back(host.size());
    fdtd::TMz::backends::gpu::download(std::span<FloatType>(back), device, stream);
    CHECK(back == host);

    device.zero(stream);
    fdtd::TMz::backends::gpu::download(std::span<FloatType>(back), device, stream);
    CHECK(back == std::vector<FloatType>(host.size(), FloatType(0)));
}

// Verify that a move leaves the source empty and the target with the memory.
TEST_CASE("Device buffer moves", "[unit][cuda]")
{
    require_device();
    DeviceBuffer<FloatType> first(16);
    const FloatType *address = first.data();
    CHECK(address != nullptr);
    CHECK(first.size() == 16);

    DeviceBuffer<FloatType> second(std::move(first));
    CHECK(first.empty());
    CHECK(first.data() == nullptr);
    CHECK(second.data() == address);
    CHECK(second.size() == 16);

    DeviceBuffer<FloatType> third;
    CHECK(third.empty());
    third = std::move(second);
    CHECK(second.empty());
    CHECK(third.data() == address);
}

// Verify that an empty buffer holds no memory and that a size past the device's throws.
TEST_CASE("Device buffer sizes", "[unit][cuda]")
{
    require_device();
    const DeviceBuffer<FloatType> none(0);
    CHECK(none.empty());
    CHECK(none.data() == nullptr);
    CHECK_THROWS_AS(DeviceBuffer<FloatType>(std::size_t(1) << 50),
                    fdtd::TMz::backends::gpu::CudaError);
    CHECK_THROWS_AS(DeviceBuffer<FloatType>(~std::size_t(0)), std::length_error);
}
