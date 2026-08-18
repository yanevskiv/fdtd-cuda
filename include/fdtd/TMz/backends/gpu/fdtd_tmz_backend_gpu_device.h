/*
 * C++ header for the queries of the CUDA device.
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

#ifndef FDTD_TMZ_BACKEND_GPU_DEVICE_H
#define FDTD_TMZ_BACKEND_GPU_DEVICE_H

#include <cstddef>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /** Namespace for what the 2D TMz CUDA backend's files share. */
            namespace gpu
            {
                /**
                 * Number of CUDA devices this process can use.
                 *
                 * A failure of the query, as with no driver, counts as no device.
                 *
                 * @return The device count.
                 */
                std::size_t device_count() noexcept;

                /**
                 * Most threads a block may hold on the current device.
                 *
                 * @return The limit, 1024 on every device since compute capability 2.0.
                 * @throws CudaError When the query fails.
                 */
                std::size_t max_threads_per_block();

                /** What the current device reports of its size, clocks and memory. */
                struct DeviceInfo
                {
                    /** Multiprocessors. */
                    std::size_t sms;

                    /** Peak clock of the multiprocessors in hertz. */
                    double clock;

                    /** Peak clock of the memory in hertz. */
                    double memory_clock;

                    /** Width of the memory bus in bits. */
                    std::size_t bus_bits;

                    /** Size of the L2 cache in bytes. */
                    std::size_t l2_bytes;

                    /** Major number of the compute capability. */
                    int major;

                    /** Minor number of the compute capability. */
                    int minor;
                };

                /**
                 * Size, clocks and memory of the current device.
                 *
                 * @return The device's report.
                 * @throws CudaError When a query fails.
                 */
                DeviceInfo device_info();

                /**
                 * Measure the memory bandwidth of the current device with a copy kernel.
                 *
                 * Copies a buffer to another, each of the given size, and counts the bytes read
                 * and written.
                 *
                 * @param bytes Size of each buffer, rounded down to 16 bytes.
                 * @param repeats Timed copies after one untimed, at least one.
                 * @return The median bandwidth in bytes per second.
                 * @throws CudaError When the device cannot hold the buffers or a call fails.
                 */
                double copy_bandwidth(std::size_t bytes, std::size_t repeats);
            }
        }
    }
}

#endif
