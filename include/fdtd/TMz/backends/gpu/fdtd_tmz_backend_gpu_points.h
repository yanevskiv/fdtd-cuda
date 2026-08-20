/*
 * C++ header for the CUDA kernels over the source and probe cells.
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

#ifndef FDTD_TMZ_BACKEND_GPU_POINTS_H
#define FDTD_TMZ_BACKEND_GPU_POINTS_H

#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_cell.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"
#include "fdtd/fdtd_float.h"

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
                 * Queue a kernel that adds one step's value to each of a list of source cells.
                 *
                 * @param stream Stream to queue on.
                 * @param fields Fields to add to.
                 * @param points Source cells; point.column picks the value in the step's row.
                 * @param count Number of cells, zero to queue nothing.
                 * @param values Step-major values, pointed at the step's row when step is null.
                 * @param step Step counted on the device, or null.
                 * @param stride Values per step.
                 */
                void queue_add_sources(const Stream &stream, const DeviceFields &fields,
                                       const DevicePoint *points, int count,
                                       const FloatType *values, const int *step, int stride);

                /**
                 * Queue a kernel that copies each of a list of probe cells into one step's
                 * records.
                 *
                 * @param stream Stream to queue on.
                 * @param fields Fields to read.
                 * @param points Probe cells; point.column picks the record in the step's row.
                 * @param count Number of cells, zero to queue nothing.
                 * @param records Step-major records, pointed at the step's row when step is
                 *        null.
                 * @param step Step counted on the device, or null.
                 * @param stride Records per step.
                 */
                void queue_record_probes(const Stream &stream, const DeviceFields &fields,
                                         const DevicePoint *points, int count, FloatType *records,
                                         const int *step, int stride);

                /**
                 * Queue a kernel that sets the step counted on the device.
                 *
                 * @param stream Stream to queue on.
                 * @param step The device counter.
                 * @param value Step to set.
                 */
                void queue_set_step(const Stream &stream, int *step, int value);

                /**
                 * Queue a kernel that counts a step on the device.
                 *
                 * @param stream Stream to queue on.
                 * @param step The device counter.
                 */
                void queue_count_step(const Stream &stream, int *step);
            }
        }
    }
}

#endif
