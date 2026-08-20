/*
 * C++ header for the cell and step of a CUDA thread.
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

#ifndef FDTD_TMZ_BACKEND_GPU_THREAD_CUH
#define FDTD_TMZ_BACKEND_GPU_THREAD_CUH

#include <cuda_runtime.h>

#include <cstddef>

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
                 * Column of the thread along x in the launch.
                 *
                 * @return blockIdx.x * blockDim.x + threadIdx.x.
                 */
                __device__ inline int thread_i()
                {
                    return static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
                }

                /**
                 * Row of the thread along y in the launch.
                 *
                 * @return blockIdx.y * blockDim.y + threadIdx.y.
                 */
                __device__ inline int thread_j()
                {
                    return static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
                }

                /**
                 * Start of a step's row in a step-major array.
                 *
                 * @param step Step counted on the device, or null for a launch already pointed at
                 *        its row.
                 * @param stride Elements per row.
                 * @return step * stride, or zero for a null step.
                 */
                __device__ inline std::ptrdiff_t step_row(const int *step, int stride)
                {
                    return step != nullptr ? static_cast<std::ptrdiff_t>(*step) * stride : 0;
                }
            }
        }
    }
}

#endif
