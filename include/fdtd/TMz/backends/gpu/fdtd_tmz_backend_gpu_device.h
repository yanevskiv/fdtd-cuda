/*
 * C++ header for the queries of the CUDA device.
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

#ifndef FDTD_TMZ_BACKEND_GPU_DEVICE_H
#define FDTD_TMZ_BACKEND_GPU_DEVICE_H

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
                 * Number of CUDA devices this process can use.
                 *
                 * A failure of the query, as with no driver, counts as no device.
                 *
                 * @return The device count.
                 */
                std::size_t device_count() noexcept;

                /**
                 * Most threads a block may hold on the current device.
                 *
                 * @return The limit, 1024 on every device since compute capability 2.0.
                 * @throws CudaError When the query fails.
                 */
                std::size_t max_threads_per_block();
            }
        }
    }
}

#endif
