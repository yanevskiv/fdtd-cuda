/*
 * C++ header for a CUDA event that marks a point in a stream.
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

#ifndef FDTD_TMZ_BACKEND_GPU_EVENT_H
#define FDTD_TMZ_BACKEND_GPU_EVENT_H

#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"

/** The CUDA runtime's event type, which cudaEvent_t points to. */
struct CUevent_st;

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
                 * A CUDA event: a point in a stream's work that the host or another stream can wait for, and that times the work between
                 * two of them.
                 *
                 * Its header holds no CUDA type but the event's opaque pointer, so host code can keep one.
                 */
                class Event
                {
                public:
                    /**
                     * Create the event.
                     *
                     * @throws CudaError When CUDA cannot create it.
                     */
                    Event();

                    /** Destroy the event, ignoring any error, since a destructor cannot throw. */
                    ~Event();

                    /** Copying is not allowed: the event has one owner. */
                    Event(const Event &) = delete;

                    /**
                     * Take the event of another, which is left with none.
                     *
                     * @param other Event to take from.
                     */
                    Event(Event &&other) noexcept;

                    /** Copying is not allowed: the event has one owner. */
                    Event &operator=(const Event &) = delete;

                    /**
                     * Destroy this event and take the event of another, which is left with none.
                     *
                     * @param other Event to take from.
                     * @return This event.
                     */
                    Event &operator=(Event &&other) noexcept;

                    /**
                     * Mark the point after the work queued on a stream so far.
                     *
                     * @param stream Stream to mark.
                     * @throws CudaError When CUDA rejects the call.
                     */
                    void record(const Stream &stream);

                    /**
                     * Make the work queued on a stream from now on wait for the marked point.
                     *
                     * @param stream Stream that waits.
                     * @throws CudaError When CUDA rejects the call.
                     */
                    void hold(const Stream &stream) const;

                    /**
                     * Wait on the host for the marked point.
                     *
                     * @throws CudaError When the work before it or the wait failed.
                     */
                    void synchronize() const;

                    /**
                     * Milliseconds from another event's point to this one's, both reached.
                     *
                     * @param start The earlier event.
                     * @return The time between them.
                     * @throws CudaError When CUDA cannot tell.
                     */
                    float milliseconds_since(const Event &start) const;

                private:
                    /** The CUDA event. */
                    CUevent_st *m_handle = nullptr;
                };
            }
        }
    }
}

#endif
