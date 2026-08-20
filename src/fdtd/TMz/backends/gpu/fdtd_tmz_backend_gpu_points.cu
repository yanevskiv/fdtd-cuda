/*
 * C++ file for the CUDA kernels over the source and probe cells.
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

// Declarations.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_points.h"

// The launch check and the thread's cell and step.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_thread.cuh"

// CUDA runtime.
#include <cuda_runtime.h>

// Namespace for what the 2D TMz CUDA backend's files share.
namespace fdtd::TMz::backends::gpu
{
    // Helpers private to this file.
    namespace
    {
        // Threads per block of the kernels over a list of cells.
        constexpr int LINE_BLOCK = 128;

        // Blocks of LINE_BLOCK threads that cover a number of cells.
        unsigned line_blocks(int count)
        {
            return static_cast<unsigned>((count + LINE_BLOCK - 1) / LINE_BLOCK);
        }

        // Add one step's value to the thread's source cell, the step on the device picking the row.
        __global__ void add_sources_kernel(DeviceFields fields, const DevicePoint *points,
                                           int count, const FloatType *values, const int *step,
                                           int stride)
        {
            const int t = thread_i();
            if (t < count) {
                const DevicePoint point = points[t];
                const FloatType value = values[step_row(step, stride) + point.column];
                field_of(fields, point.component)[point.cell] += value;
            }
        }

        // Copy the thread's probe cell into one step's records, the step picking the row.
        __global__ void record_probes_kernel(DeviceFields fields, const DevicePoint *points,
                                             int count, FloatType *records, const int *step,
                                             int stride)
        {
            const int t = thread_i();
            if (t < count) {
                const DevicePoint point = points[t];
                records[step_row(step, stride) + point.column] =
                    field_of(fields, point.component)[point.cell];
            }
        }

        // Set the step on the device.
        __global__ void set_step_kernel(int *step, int value)
        {
            *step = value;
        }

        // Count a step on the device.
        __global__ void count_step_kernel(int *step)
        {
            ++*step;
        }
    }

    // Launch one thread per source cell.
    void queue_add_sources(const Stream &stream, const DeviceFields &fields,
                           const DevicePoint *points, int count, const FloatType *values,
                           const int *step, int stride)
    {
        if (count == 0) {
            return;
        }
        add_sources_kernel<<<line_blocks(count), LINE_BLOCK, 0, stream.handle()>>>(
            fields, points, count, values, step, stride);
        check(cudaGetLastError(), "kernel launch");
    }

    // Launch one thread per probe cell.
    void queue_record_probes(const Stream &stream, const DeviceFields &fields,
                             const DevicePoint *points, int count, FloatType *records,
                             const int *step, int stride)
    {
        if (count == 0) {
            return;
        }
        record_probes_kernel<<<line_blocks(count), LINE_BLOCK, 0, stream.handle()>>>(
            fields, points, count, records, step, stride);
        check(cudaGetLastError(), "kernel launch");
    }

    // Launch one thread to set the counter.
    void queue_set_step(const Stream &stream, int *step, int value)
    {
        set_step_kernel<<<1, 1, 0, stream.handle()>>>(step, value);
        check(cudaGetLastError(), "kernel launch");
    }

    // Launch one thread to count.
    void queue_count_step(const Stream &stream, int *step)
    {
        count_step_kernel<<<1, 1, 0, stream.handle()>>>(step);
        check(cudaGetLastError(), "kernel launch");
    }
}
