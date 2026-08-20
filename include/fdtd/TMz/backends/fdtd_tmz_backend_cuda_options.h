/*
 * C++ header for the settings of the 2D TMz CUDA backend and the variant names they read from.
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

#ifndef FDTD_TMZ_BACKEND_CUDA_OPTIONS_H
#define FDTD_TMZ_BACKEND_CUDA_OPTIONS_H

#include <cstddef>
#include <string>
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
            /** Optimization level of the CUDA backend, each the one before with one change. */
            enum class CudaLevel {
                /** Two kernels: an H kernel, then an E kernel, one thread per cell, eight
                 * coefficient arrays. */
                V1,

                /** Graph: v1 with its steps replayed from a captured CUDA graph. */
                V2,

                /** Table: v2 with the cells outside the PML reading a table of the distinct
                 * coefficient tuples, from the cell's position when the interior shares one
                 * tuple and from a byte per cell otherwise. */
                V3,

                /** Tiled: v3 with one kernel per step, each block keeping the H of its tile and
                 * the tile's lower and left halo in shared memory. The default. */
                V4
            };

            /**
             * Settings of the CUDA backend, each with a usable default.
             *
             * A variant name, `<level>[:<BX>x<BY>][+async]`, sets every field. The level is `v1`
             * to `v4`; `:BXxBY` sets the thread block, the default one when it is left out;
             * `+async` copies snapshots out while the steps go on. `default` names the default
             * settings. Whether the device takes the block is left to the backend.
             */
            struct CudaOptions
            {
                /** Optimization level. */
                CudaLevel level = CudaLevel::V4;

                /** Threads along x of a thread block, positive. */
                std::size_t block_x = 32;

                /** Threads along y of a thread block, positive, with block_x * block_y within the
                 * device's threads per block. */
                std::size_t block_y = 8;

                /** Whether the backend copies the snapshots of a run out while it goes on
                 * stepping, instead of stopping for each. */
                bool async_snapshots = false;

                /**
                 * Read the settings of a variant name.
                 *
                 * @param variant Variant name, as `v2:16x16+async`, or `default`.
                 * @return The settings.
                 * @throws std::invalid_argument When the name is malformed, with the reason and
                 *         the name in the message.
                 */
                static CudaOptions parse(std::string_view variant);

                /**
                 * Canonical variant name of the settings, which parse() reads back.
                 *
                 * The block is always written.
                 *
                 * @return The name, as `v1:32x8` or `v4:16x16+async`.
                 */
                std::string name() const;
            };
        }
    }
}

#endif
