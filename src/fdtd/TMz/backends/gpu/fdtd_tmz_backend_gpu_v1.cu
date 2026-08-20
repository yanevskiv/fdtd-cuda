/*
 * C++ file for the 2D TMz CUDA backend's level v1, two kernels.
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

// The launch check, the thread's cell and the kernels over the source cells.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_points.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_thread.cuh"

// CUDA runtime.
#include <cuda_runtime.h>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Helpers private to this file.
    namespace
    {
        // Advance Hx and Hy of the thread's cell.
        __global__ void update_h_kernel(gpu::DeviceFields fields, gpu::DeviceCoeffs coeffs)
        {
            const int i = gpu::thread_i();
            const int j = gpu::thread_j();
            if (i < fields.nx && j < fields.ny) {
                gpu::update_h_cell(fields, coeffs, i, j);
            }
        }

        // Advance Ez of the thread's cell by the edge, PML or interior update.
        __global__ void update_e_kernel(gpu::DeviceFields fields, gpu::DeviceCoeffs coeffs,
                                        Boundary boundary)
        {
            const int i = gpu::thread_i();
            const int j = gpu::thread_j();
            if (i < fields.nx && j < fields.ny) {
                gpu::update_e_cell(fields, coeffs, i, j, boundary);
            }
        }
    }

    // Cover the grid with blocks, one thread per cell, and queue H, its sources, then E.
    gpu::DeviceFields CudaBackend::launch_two_kernels(const gpu::DeviceCoeffs &coeffs,
                                                      const StepPoints &points)
    {
        const gpu::DeviceFields fields = device_fields();
        const auto bx = static_cast<unsigned>(m_options.block_x);
        const auto by = static_cast<unsigned>(m_options.block_y);
        const dim3 block(bx, by);
        const dim3 cells(static_cast<unsigned>((m_grid.nx() + bx - 1) / bx),
                         static_cast<unsigned>((m_grid.ny() + by - 1) / by));
        cudaStream_t stream = m_stream.handle();
        update_h_kernel<<<cells, block, 0, stream>>>(fields, coeffs);
        gpu::check_launch();
        gpu::queue_add_sources(m_stream, fields, points.sources, points.magnetic, points.values,
                               points.step, points.magnetic + points.electric);
        update_e_kernel<<<cells, block, 0, stream>>>(fields, coeffs, m_boundary);
        gpu::check_launch();
        return fields;
    }
}
