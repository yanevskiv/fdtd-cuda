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

// CUDA runtime.
#include <cuda_runtime.h>

// Standard library for sizes.
#include <cstddef>

// Namespace for what the 2D TMz CUDA backend's files share.
namespace fdtd::TMz::backends::gpu
{
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
}
