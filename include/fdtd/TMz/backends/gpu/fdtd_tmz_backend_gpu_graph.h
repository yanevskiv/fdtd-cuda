/*
 * C++ header for a CUDA graph of captured work, replayed on a stream.
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

#ifndef FDTD_TMZ_BACKEND_GPU_GRAPH_H
#define FDTD_TMZ_BACKEND_GPU_GRAPH_H

#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"

/** The CUDA runtime's executable graph type, which cudaGraphExec_t points to. */
struct CUgraphExec_st;

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
                 * An executable CUDA graph: work captured once from a stream, then replayed on it with one launch.
                 *
                 * Its header holds no CUDA type but the graph's opaque pointer, so host code can keep one. A replay runs the captured
                 * kernels with the arguments they had when captured.
                 */
                class Graph
                {
                public:
                    /** Create an empty graph. */
                    Graph() noexcept = default;

                    /** Destroy the graph, ignoring any error, since a destructor cannot throw. */
                    ~Graph();

                    /** Copying is not allowed: the graph has one owner. */
                    Graph(const Graph &) = delete;

                    /**
                     * Take the graph of another, which is left empty.
                     *
                     * @param other Graph to take from.
                     */
                    Graph(Graph &&other) noexcept;

                    /** Copying is not allowed: the graph has one owner. */
                    Graph &operator=(const Graph &) = delete;

                    /**
                     * Destroy this graph and take the graph of another, which is left empty.
                     *
                     * @param other Graph to take from.
                     * @return This graph.
                     */
                    Graph &operator=(Graph &&other) noexcept;

                    /**
                     * Capture the work a callable queues on a stream, in place of any graph held; the work is recorded, not run.
                     *
                     * @param stream Stream the callable queues on.
                     * @param queue Callable that queues the work.
                     * @throws CudaError When the capture or the instantiation fails.
                     */
                    template <typename F> void capture(const Stream &stream, F &&queue)
                    {
                        begin_capture(stream);
                        try {
                            queue();
                        } catch (...) {
                            cancel_capture(stream);
                            throw;
                        }
                        end_capture(stream);
                    }

                    /**
                     * Queue a replay of the graph on a stream.
                     *
                     * @param stream Stream to replay on.
                     * @throws CudaError When the launch fails.
                     */
                    void launch(const Stream &stream) const;

                    /**
                     * Whether no graph is held.
                     *
                     * @return True before the first capture and after a move.
                     */
                    bool empty() const noexcept;

                private:
                    /**
                     * Start capturing the work queued on a stream by this thread.
                     *
                     * @param stream Stream to capture.
                     * @throws CudaError When the capture cannot start.
                     */
                    void begin_capture(const Stream &stream);

                    /**
                     * Stop capturing and make the captured work this graph.
                     *
                     * @param stream Stream being captured.
                     * @throws CudaError When the capture or the instantiation fails.
                     */
                    void end_capture(const Stream &stream);

                    /**
                     * Stop capturing and drop the captured work, ignoring any error.
                     *
                     * @param stream Stream being captured.
                     */
                    static void cancel_capture(const Stream &stream) noexcept;

                    /** The executable graph. */
                    CUgraphExec_st *m_exec = nullptr;
                };
            }
        }
    }
}

#endif
