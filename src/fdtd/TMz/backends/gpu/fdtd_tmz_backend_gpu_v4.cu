/*
 * C++ file for the 2D TMz CUDA backend's level v4, tiled.
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

// The launch check and the thread's cell and step.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_thread.cuh"

// CUDA runtime.
#include <cuda_runtime.h>

// Standard library for sizes and swaps.
#include <cstddef>
#include <utility>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Helpers private to this file.
    namespace
    {
        // Add the magnetic sources on a cell to its new H, as the source kernel would.
        __device__ inline void add_h_sources(gpu::HValues &h, int k, const gpu::DevicePoint *points,
                                             int count, const FloatType *values, const int *step,
                                             int stride)
        {
            for (int s = 0; s < count; ++s) {
                const gpu::DevicePoint point = points[s];
                if (point.cell == k) {
                    const FloatType value = values[gpu::step_row(step, stride) + point.column];
                    if (point.component == Component::Hx) {
                        h.hx += value;
                    } else {
                        h.hy += value;
                    }
                }
            }
        }

        // Advance H and E of the thread's cell in one pass, with the tile's H in shared memory:
        // each thread computes its cell's new H, the first column and row of the block also the
        // H of the cell to their left and below, then, after a barrier, Ez from the tile.
        __global__ void update_tiled_kernel(gpu::DeviceFields fields, gpu::DeviceCoeffs coeffs,
                                            Boundary boundary, const gpu::DevicePoint *points,
                                            int magnetic, const FloatType *values, const int *step,
                                            int stride)
        {
            extern __shared__ FloatType tile[];
            const int width = static_cast<int>(blockDim.x) + 1;
            FloatType *tileHx = tile;
            FloatType *tileHy = tile + width * (static_cast<int>(blockDim.y) + 1);
            const int c =
                (static_cast<int>(threadIdx.y) + 1) * width + static_cast<int>(threadIdx.x) + 1;
            const int i = gpu::thread_i();
            const int j = gpu::thread_j();
            const bool inside = i < fields.nx && j < fields.ny;
            const int k = j * fields.pitch + i;
            if (inside) {
                gpu::HValues h = gpu::h_values(fields, coeffs, i, j);
                add_h_sources(h, k, points, magnetic, values, step, stride);
                tileHx[c] = h.hx;
                tileHy[c] = h.hy;
                fields.hx_next[k] = h.hx;
                fields.hy_next[k] = h.hy;
                if (threadIdx.x == 0 && i > 0) {
                    gpu::HValues left = gpu::h_values(fields, coeffs, i - 1, j);
                    add_h_sources(left, k - 1, points, magnetic, values, step, stride);
                    tileHy[c - 1] = left.hy;
                }
                if (threadIdx.y == 0 && j > 0) {
                    gpu::HValues below = gpu::h_values(fields, coeffs, i, j - 1);
                    add_h_sources(below, k - fields.pitch, points, magnetic, values, step, stride);
                    tileHx[c - width] = below.hx;
                }
            }
            __syncthreads();
            if (inside) {
                const gpu::HAround h{
                    .hy = i + 1 < fields.nx ? tileHy[c] : FloatType(0),
                    .hy_left = i > 0 ? tileHy[c - 1] : FloatType(0),
                    .hx = j + 1 < fields.ny ? tileHx[c] : FloatType(0),
                    .hx_below = j > 0 ? tileHx[c - width] : FloatType(0),
                };
                fields.ez_next[k] = gpu::e_value(fields, coeffs, i, j, boundary, h);
            }
        }
    }

    // Cover the grid with blocks, queue the one kernel with room for the tile and its halo, and
    // swap the arrays it read with those it wrote.
    gpu::DeviceFields CudaBackend::launch_tiled(const gpu::DeviceCoeffs &coeffs,
                                                const StepPoints &points)
    {
        const gpu::DeviceFields fields = device_fields();
        const auto bx = static_cast<unsigned>(m_options.block_x);
        const auto by = static_cast<unsigned>(m_options.block_y);
        const dim3 block(bx, by);
        const dim3 cells(static_cast<unsigned>((m_grid.nx() + bx - 1) / bx),
                         static_cast<unsigned>((m_grid.ny() + by - 1) / by));
        const std::size_t shared = 2 * (bx + 1) * (by + 1) * sizeof(FloatType);
        update_tiled_kernel<<<cells, block, shared, m_stream.handle()>>>(
            fields, coeffs, m_boundary, points.sources, points.magnetic, points.values, points.step,
            points.magnetic + points.electric);
        gpu::check_launch();
        std::swap(m_ez, m_ez_next);
        std::swap(m_hx, m_hx_next);
        std::swap(m_hy, m_hy_next);
        return device_fields();
    }
}
