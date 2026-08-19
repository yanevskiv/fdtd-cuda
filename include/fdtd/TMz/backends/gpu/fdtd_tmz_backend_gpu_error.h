/*
 * C++ header for the error a failed CUDA call throws.
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

#ifndef FDTD_TMZ_BACKEND_GPU_ERROR_H
#define FDTD_TMZ_BACKEND_GPU_ERROR_H

#include <stdexcept>
#include <string>

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
                 * Error of a failed CUDA call, with the CUDA error code.
                 *
                 * Code outside the CUDA backend sees it as the std::runtime_error it derives from.
                 */
                class CudaError : public std::runtime_error
                {
                public:
                    /**
                     * Create the error.
                     *
                     * @param code CUDA error code, as a cudaError_t converted to int.
                     * @param message What failed, naming the call.
                     */
                    CudaError(int code, const std::string &message);

                    /**
                     * CUDA error code of the failed call.
                     *
                     * @return The code, as a cudaError_t converted to int.
                     */
                    int code() const noexcept;

                private:
                    /** CUDA error code. */
                    int m_code;
                };
            }
        }
    }
}

#endif
