/*
 * C++ file for a CUDA event that marks a point in a stream.
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
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_event.h"

// The check that turns a failed call into a CudaError.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"

// CUDA runtime.
#include <cuda_runtime.h>

// Standard library for exchanges.
#include <utility>

// Namespace for what the 2D TMz CUDA backend's files share.
namespace fdtd::TMz::backends::gpu
{
    // Create the event.
    Event::Event()
    {
        check(cudaEventCreate(&m_handle), "cudaEventCreate");
    }

    // Destroy the event, if any.
    Event::~Event()
    {
        if (m_handle != nullptr) {
            cudaEventDestroy(m_handle);
        }
    }

    // Take the other's event.
    Event::Event(Event &&other) noexcept
        : m_handle(std::exchange(other.m_handle, nullptr))
    {
        // Empty.
    }

    // Destroy this event and take the other's.
    Event &Event::operator=(Event &&other) noexcept
    {
        if (this != &other) {
            if (m_handle != nullptr) {
                cudaEventDestroy(m_handle);
            }
            m_handle = std::exchange(other.m_handle, nullptr);
        }
        return *this;
    }

    // Record the event on the stream.
    void Event::record(const Stream &stream)
    {
        check(cudaEventRecord(m_handle, stream.handle()), "cudaEventRecord");
    }

    // Make the stream wait for the event.
    void Event::hold(const Stream &stream) const
    {
        check(cudaStreamWaitEvent(stream.handle(), m_handle, 0), "cudaStreamWaitEvent");
    }

    // Wait for the event on the host.
    void Event::synchronize() const
    {
        check(cudaEventSynchronize(m_handle), "cudaEventSynchronize");
    }

    // Ask CUDA for the time between the events.
    float Event::milliseconds_since(const Event &start) const
    {
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start.m_handle, m_handle), "cudaEventElapsedTime");
        return ms;
    }
}
