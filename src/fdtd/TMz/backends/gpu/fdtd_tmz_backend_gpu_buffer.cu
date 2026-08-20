/*
 * C++ file for buffers in CUDA device memory and the copies to and from them.
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

// Function declarations.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_buffer.h"

// The check that turns a failed call into a CudaError.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"

// CUDA runtime.
#include <cuda_runtime.h>

// Standard library for sizes, limits and errors.
#include <cstddef>
#include <limits>
#include <stdexcept>

// Namespace for what the 2D TMz CUDA backend's files share.
namespace fdtd::TMz::backends::gpu
{
    // Helpers private to this file.
    namespace
    {
        // Bytes a row is padded to.
        constexpr std::size_t ROW_ALIGNMENT = 128;
    }

    // Allocate the bytes of count elements, none for an empty array.
    void *allocate(std::size_t count, std::size_t size)
    {
        if (count == 0 || size == 0) {
            return nullptr;
        }
        if (count > std::numeric_limits<std::size_t>::max() / size) {
            throw std::length_error("DeviceBuffer: size in bytes overflows");
        }
        void *pointer = nullptr;
        check(cudaMalloc(&pointer, count * size), "cudaMalloc");
        return pointer;
    }

    // Free the memory, ignoring any error.
    void release(void *pointer) noexcept
    {
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
    }

    // Round the bytes of a row up to the alignment, in elements.
    std::size_t device_pitch(std::size_t width, std::size_t size) noexcept
    {
        const std::size_t perRow = ROW_ALIGNMENT / size;
        return (width + perRow - 1) / perRow * perRow;
    }

    // Queue the host-to-device copy of the rows.
    void copy_rows_to_device(void *target, std::size_t targetPitch, const void *source,
                             std::size_t sourcePitch, std::size_t width, std::size_t height,
                             const Stream &stream)
    {
        if (width == 0 || height == 0) {
            return;
        }
        check(cudaMemcpy2DAsync(target, targetPitch, source, sourcePitch, width, height,
                                cudaMemcpyHostToDevice, stream.handle()),
              "cudaMemcpy2DAsync");
    }

    // Copy the rows to the host and wait for them.
    void copy_rows_to_host(void *target, std::size_t targetPitch, const void *source,
                           std::size_t sourcePitch, std::size_t width, std::size_t height,
                           const Stream &stream)
    {
        if (width == 0 || height == 0) {
            return;
        }
        check(cudaMemcpy2DAsync(target, targetPitch, source, sourcePitch, width, height,
                                cudaMemcpyDeviceToHost, stream.handle()),
              "cudaMemcpy2DAsync");
        stream.synchronize();
    }

    // Queue the device-to-host copy of the rows.
    void queue_rows_to_host(void *target, std::size_t targetPitch, const void *source,
                            std::size_t sourcePitch, std::size_t width, std::size_t height,
                            const Stream &stream)
    {
        if (width == 0 || height == 0) {
            return;
        }
        check(cudaMemcpy2DAsync(target, targetPitch, source, sourcePitch, width, height,
                                cudaMemcpyDeviceToHost, stream.handle()),
              "cudaMemcpy2DAsync");
    }

    // Queue the device-to-device copy of the bytes.
    void copy_on_device(void *target, const void *source, std::size_t bytes, const Stream &stream)
    {
        if (bytes == 0) {
            return;
        }
        check(cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToDevice, stream.handle()),
              "cudaMemcpyAsync");
    }

    // Queue the zeroing of the bytes.
    void fill_zero(void *target, std::size_t bytes, const Stream &stream)
    {
        if (bytes == 0) {
            return;
        }
        check(cudaMemsetAsync(target, 0, bytes, stream.handle()), "cudaMemsetAsync");
    }
}
