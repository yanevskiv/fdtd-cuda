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
            /** How the CUDA backend lays the update of a step out in kernels. */
            enum class CudaKernel {
                /** One E kernel for every cell, each thread taking its cell's edge, PML or
                 * interior branch. */
                Fused,

                /** One kernel per step: each block updates H on its tile and the tile's lower and
                 * left halo, keeps it in shared memory, and updates E from it, reading Ez, Hx and
                 * Hy once and writing them to a second set of arrays. The default. */
                Tiled
            };

            /** Where the CUDA kernels read the update coefficients of a cell. */
            enum class CudaCoeffs {
                /** Eight arrays with a value per cell. */
                Arrays,

                /** A table of the distinct tuples of the cells outside the PML, read from the
                 * cell's position when the interior shares one tuple and from a byte per cell
                 * otherwise; the PML cells read the arrays. The default. */
                Table
            };

            /**
             * Settings of the CUDA backend, each with a usable default.
             *
             * A variant name, `<kernel>[:<BX>x<BY>][+table][+graph][+async]`, sets every field.
             * The kernel is `fused` or `tiled`; `:BXxBY` sets the thread block, the default one
             * when it is left out; `+table` reads the coefficients outside the PML from a table;
             * `+graph` replays captured steps as a CUDA graph; `+async` copies snapshots out
             * while the steps go on. The suffixes may come in any order. `default` names the
             * default settings. Whether the device takes the block is left to the backend.
             */
            struct CudaOptions
            {
                /** How a step is laid out in kernels. */
                CudaKernel kernel = CudaKernel::Tiled;

                /** Threads along x of a thread block, positive. */
                std::size_t block_x = 32;

                /** Threads along y of a thread block, positive, with block_x * block_y within the
                 * device's threads per block. */
                std::size_t block_y = 8;

                /** Where the kernels read the coefficients. */
                CudaCoeffs coeffs = CudaCoeffs::Table;

                /** Whether the backend replays a graph of 16 captured steps instead of launching
                 * each kernel of each step. */
                bool graph = true;

                /** Whether the backend copies the snapshots of a run out while it goes on
                 * stepping, instead of stopping for each. */
                bool async_snapshots = false;

                /**
                 * Read the settings of a variant name.
                 *
                 * @param variant Variant name, as `tiled:16x16+table`, or `default`.
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
                 * @return The name, as `fused:32x8` or `tiled:16x16+table+graph+async`.
                 */
                std::string name() const;
            };
        }
    }
}

#endif
