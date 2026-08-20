/*
 * C++ file for the CUDA stream the backend's work is queued on.
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
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"

// The check that turns a failed call into a CudaError.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"

// CUDA runtime.
#include <cuda_runtime.h>

// Standard library for exchanges.
#include <utility>

// Namespace for what the 2D TMz CUDA backend's files share.
namespace fdtd::TMz::backends::gpu
{
    // Create a stream that does not wait for the legacy default stream.
    Stream::Stream()
    {
        check(cudaStreamCreateWithFlags(&m_handle, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
    }

    // Destroy the stream, if any.
    Stream::~Stream()
    {
        if (m_handle != nullptr) {
            cudaStreamDestroy(m_handle);
        }
    }

    // Take the other's stream.
    Stream::Stream(Stream &&other) noexcept
        : m_handle(std::exchange(other.m_handle, nullptr))
    {
        // Empty.
    }

    // Destroy this stream and take the other's.
    Stream &Stream::operator=(Stream &&other) noexcept
    {
        if (this != &other) {
            if (m_handle != nullptr) {
                cudaStreamDestroy(m_handle);
            }
            m_handle = std::exchange(other.m_handle, nullptr);
        }
        return *this;
    }

    // Wait for the queued work.
    void Stream::synchronize() const
    {
        check(cudaStreamSynchronize(m_handle), "cudaStreamSynchronize");
    }

    // Return the stream.
    CUstream_st *Stream::handle() const noexcept
    {
        return m_handle;
    }
}
