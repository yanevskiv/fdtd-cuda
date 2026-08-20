/*
 * C++ header for the check that turns a failed CUDA call into a CudaError.
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

#ifndef FDTD_TMZ_BACKEND_GPU_CHECK_CUH
#define FDTD_TMZ_BACKEND_GPU_CHECK_CUH

#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_error.h"

#include <cuda_runtime.h>

#include <sstream>
#include <string_view>

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
                 * Throw a CudaError when a CUDA call failed.
                 *
                 * The message is the call's name, then CUDA's description of the error.
                 *
                 * @param status Status the call returned.
                 * @param call Name of the call, as "cudaMalloc".
                 * @throws CudaError When the status is not cudaSuccess.
                 */
                inline void check(cudaError_t status, std::string_view call)
                {
                    if (status == cudaSuccess) {
                        return;
                    }
                    std::ostringstream message;
                    message << call;
                    message << ": ";
                    message << cudaGetErrorString(status);
                    throw CudaError(static_cast<int>(status), message.str());
                }

                /**
                 * Throw a CudaError when a kernel launch queued since the last check failed.
                 *
                 * @throws CudaError When cudaGetLastError() reports an error.
                 */
                inline void check_launch()
                {
                    check(cudaGetLastError(), "kernel launch");
                }
            }
        }
    }
}

#endif
