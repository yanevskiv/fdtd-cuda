/*
 * C++ header for the CUDA stream the backend's work is queued on.
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

#ifndef FDTD_TMZ_BACKEND_GPU_STREAM_H
#define FDTD_TMZ_BACKEND_GPU_STREAM_H

/** The CUDA runtime's stream type, which cudaStream_t points to. */
struct CUstream_st;

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
                 * A CUDA stream that does not wait for the legacy default stream.
                 *
                 * Its header holds no CUDA type but the stream's opaque pointer, so host code can keep one. Work queued on it runs in
                 * order; synchronize() waits for it and reports the first error.
                 */
                class Stream
                {
                public:
                    /**
                     * Create the stream.
                     *
                     * @throws CudaError When CUDA cannot create it.
                     */
                    Stream();

                    /** Destroy the stream, ignoring any error, since a destructor cannot throw. */
                    ~Stream();

                    /** Copying is not allowed: the stream has one owner. */
                    Stream(const Stream &) = delete;

                    /**
                     * Take the stream of another, which is left with none.
                     *
                     * @param other Stream to take from.
                     */
                    Stream(Stream &&other) noexcept;

                    /** Copying is not allowed: the stream has one owner. */
                    Stream &operator=(const Stream &) = delete;

                    /**
                     * Destroy this stream and take the stream of another, which is left with none.
                     *
                     * @param other Stream to take from.
                     * @return This stream.
                     */
                    Stream &operator=(Stream &&other) noexcept;

                    /**
                     * Wait for the work queued so far.
                     *
                     * @throws CudaError When the work or the wait failed.
                     */
                    void synchronize() const;

                    /**
                     * The CUDA stream, for calls that take a cudaStream_t.
                     *
                     * @return The stream, null after a move.
                     */
                    CUstream_st *handle() const noexcept;

                private:
                    /** The CUDA stream. */
                    CUstream_st *m_handle = nullptr;
                };
            }
        }
    }
}

#endif
