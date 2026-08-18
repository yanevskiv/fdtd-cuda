/*
 * C++ file for a CUDA graph of captured work, replayed on a stream.
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
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_graph.h"

// The check that turns a failed call into a CudaError.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"

// CUDA runtime.
#include <cuda_runtime.h>

// Standard library for exchanges.
#include <utility>

// Namespace for what the 2D TMz CUDA backend's files share.
namespace fdtd::TMz::backends::gpu
{
    // Destroy the graph, if any.
    Graph::~Graph()
    {
        if (m_exec != nullptr) {
            cudaGraphExecDestroy(m_exec);
        }
    }

    // Take the other's graph.
    Graph::Graph(Graph &&other) noexcept
        : m_exec(std::exchange(other.m_exec, nullptr))
    {
        // Empty.
    }

    // Destroy this graph and take the other's.
    Graph &Graph::operator=(Graph &&other) noexcept
    {
        if (this != &other) {
            if (m_exec != nullptr) {
                cudaGraphExecDestroy(m_exec);
            }
            m_exec = std::exchange(other.m_exec, nullptr);
        }
        return *this;
    }

    // Replay the graph.
    void Graph::launch(const Stream &stream) const
    {
        check(cudaGraphLaunch(m_exec, stream.handle()), "cudaGraphLaunch");
    }

    // Report whether a graph is held.
    bool Graph::empty() const noexcept
    {
        return m_exec == nullptr;
    }

    // Capture only what this thread queues, so other threads' CUDA calls are not disturbed.
    void Graph::begin_capture(const Stream &stream)
    {
        check(cudaStreamBeginCapture(stream.handle(), cudaStreamCaptureModeThreadLocal),
              "cudaStreamBeginCapture");
    }

    // End the capture, instantiate it, and replace the graph held.
    void Graph::end_capture(const Stream &stream)
    {
        cudaGraph_t graph = nullptr;
        check(cudaStreamEndCapture(stream.handle(), &graph), "cudaStreamEndCapture");
        cudaGraphExec_t exec = nullptr;
        const cudaError_t status = cudaGraphInstantiate(&exec, graph, 0);
        cudaGraphDestroy(graph);
        check(status, "cudaGraphInstantiate");
        if (m_exec != nullptr) {
            cudaGraphExecDestroy(m_exec);
        }
        m_exec = exec;
    }

    // End the capture and destroy what it holds.
    void Graph::cancel_capture(const Stream &stream) noexcept
    {
        cudaGraph_t graph = nullptr;
        if (cudaStreamEndCapture(stream.handle(), &graph) == cudaSuccess && graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        cudaGetLastError();
    }
}
