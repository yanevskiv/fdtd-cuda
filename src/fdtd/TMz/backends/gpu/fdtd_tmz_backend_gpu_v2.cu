/*
 * C++ file for the 2D TMz CUDA backend's level v2, graph.
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
#include "fdtd/TMz/backends/fdtd_tmz_backend_cuda.h"

// The kernel that sets the step counted on the device.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_points.h"

// Standard library for sizes.
#include <cstddef>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Capture GRAPH_STEPS steps that read the step from the device counter, once per key, then
    // set the counter and replay the graph over each whole run of GRAPH_STEPS steps.
    std::size_t CudaBackend::queue_graph(const gpu::DeviceCoeffs &coeffs, std::size_t from,
                                         std::size_t to, std::size_t magnetic, std::size_t electric)
    {
        if (to - from < GRAPH_STEPS) {
            return from;
        }
        const GraphKey key{
            .magnetic = magnetic,
            .electric = electric,
            .probes = m_probes.size(),
            .boundary = m_boundary,
            .pml = device_fields().pml,
            .ez = m_ez.data(),
            .sources = m_device_sources.data(),
            .values = m_device_values.data(),
            .points = m_device_probes.data(),
            .records = m_device_records.data(),
        };
        if (m_graph.empty() || !(key == m_graph_key)) {
            m_graph.capture(m_stream, [&] {
                for (std::size_t s = 0; s < GRAPH_STEPS; ++s) {
                    launch_step(coeffs, 0, magnetic, electric, Launch::Replay);
                }
            });
            m_graph_key = key;
        }
        gpu::queue_set_step(m_stream, m_device_step.data(), static_cast<int>(from));
        std::size_t n = from;
        for (; n + GRAPH_STEPS <= to; n += GRAPH_STEPS) {
            m_graph.launch(m_stream);
        }
        return n;
    }
}
