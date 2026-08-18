/*
 * C++ file for host memory page-locked for the device to copy to and from on its own.
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

// Class declaration.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_pinned.h"

// The check that turns a failed call into a CudaError.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"

// CUDA runtime.
#include <cuda_runtime.h>

// Standard library for sizes and exchanges.
#include <cstddef>
#include <utility>

// Namespace for what the 2D TMz CUDA backend's files share.
namespace fdtd::TMz::backends::gpu
{
    // Register the range with CUDA.
    HostPin::HostPin(void *memory, std::size_t bytes)
    {
        if (memory == nullptr || bytes == 0) {
            return;
        }
        check(cudaHostRegister(memory, bytes, cudaHostRegisterDefault), "cudaHostRegister");
        m_memory = memory;
    }

    // Unregister the range, if any.
    HostPin::~HostPin()
    {
        if (m_memory != nullptr) {
            cudaHostUnregister(m_memory);
        }
    }

    // Take the other's range.
    HostPin::HostPin(HostPin &&other) noexcept
        : m_memory(std::exchange(other.m_memory, nullptr))
    {
        // Empty.
    }

    // Unregister this range and take the other's.
    HostPin &HostPin::operator=(HostPin &&other) noexcept
    {
        if (this != &other) {
            if (m_memory != nullptr) {
                cudaHostUnregister(m_memory);
            }
            m_memory = std::exchange(other.m_memory, nullptr);
        }
        return *this;
    }
}
