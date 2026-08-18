/*
 * C++ file for the queries of the CUDA device.
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
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_device.h"

// The check that turns a failed call into a CudaError.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"

// CUDA runtime.
#include <cuda_runtime.h>

// Device buffers and streams.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_buffer.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"

// Standard library for sizes, sorting and lists.
#include <algorithm>
#include <cstddef>
#include <vector>

// Namespace for what the 2D TMz CUDA backend's files share.
namespace fdtd::TMz::backends::gpu
{
    // Helpers private to this file.
    namespace
    {
        // Threads per block of the copy kernel.
        constexpr int COPY_BLOCK = 256;

        // Copy n vectors of four floats, each thread striding over the grid.
        __global__ void copy_kernel(const float4 *in, float4 *out, std::size_t n)
        {
            const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
            for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 k < n; k += stride) {
                out[k] = in[k];
            }
        }

        // Read one attribute of the current device.
        int attribute(cudaDeviceAttr attr)
        {
            int device = 0;
            check(cudaGetDevice(&device), "cudaGetDevice");
            int value = 0;
            check(cudaDeviceGetAttribute(&value, attr, device), "cudaDeviceGetAttribute");
            return value;
        }

        // An event that is destroyed with its scope.
        class Event
        {
        public:
            // Create the event.
            Event()
            {
                check(cudaEventCreate(&m_event), "cudaEventCreate");
            }

            // Destroy the event, ignoring errors.
            ~Event()
            {
                cudaEventDestroy(m_event);
            }

            // Events are not copied.
            Event(const Event &) = delete;

            // Events are not copied.
            Event &operator=(const Event &) = delete;

            // The event.
            cudaEvent_t get() const noexcept
            {
                return m_event;
            }

        private:
            // The event.
            cudaEvent_t m_event = nullptr;
        };
    }

    // Count the devices, clearing the error a failed query leaves behind.
    std::size_t device_count() noexcept
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        return static_cast<std::size_t>(count);
    }

    // Ask the current device for its limit.
    std::size_t max_threads_per_block()
    {
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        int threads = 0;
        check(cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, device),
              "cudaDeviceGetAttribute");
        return static_cast<std::size_t>(threads);
    }

    // Read the attributes, the clocks given in kilohertz.
    DeviceInfo device_info()
    {
        return {
            .sms = static_cast<std::size_t>(attribute(cudaDevAttrMultiProcessorCount)),
            .clock = 1e3 * attribute(cudaDevAttrClockRate),
            .memory_clock = 1e3 * attribute(cudaDevAttrMemoryClockRate),
            .bus_bits = static_cast<std::size_t>(attribute(cudaDevAttrGlobalMemoryBusWidth)),
            .l2_bytes = static_cast<std::size_t>(attribute(cudaDevAttrL2CacheSize)),
            .major = attribute(cudaDevAttrComputeCapabilityMajor),
            .minor = attribute(cudaDevAttrComputeCapabilityMinor),
        };
    }

    // Time each copy between two events and take the median.
    double copy_bandwidth(std::size_t bytes, std::size_t repeats)
    {
        const std::size_t n = bytes / sizeof(float4);
        DeviceBuffer<float4> in(n);
        DeviceBuffer<float4> out(n);
        Stream stream;
        in.zero(stream);
        const int blocks = 8 * attribute(cudaDevAttrMultiProcessorCount);

        Event start;
        Event stop;
        std::vector<double> rates;
        for (std::size_t r = 0; r <= repeats; ++r) {
            check(cudaEventRecord(start.get(), stream.handle()), "cudaEventRecord");
            copy_kernel<<<blocks, COPY_BLOCK, 0, stream.handle()>>>(in.data(), out.data(), n);
            check(cudaGetLastError(), "copy_kernel");
            check(cudaEventRecord(stop.get(), stream.handle()), "cudaEventRecord");
            check(cudaEventSynchronize(stop.get()), "cudaEventSynchronize");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, start.get(), stop.get()), "cudaEventElapsedTime");
            if (r > 0) {
                rates.push_back(2.0 * static_cast<double>(n * sizeof(float4)) / (1e-3 * ms));
            }
        }
        std::sort(rates.begin(), rates.end());
        return rates[rates.size() / 2];
    }
}
