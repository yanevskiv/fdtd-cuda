/*
 * C++ header for the options of the 2D TMz OpenMP backend.
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

#ifndef FDTD_TMZ_OMP_OPTIONS_H
#define FDTD_TMZ_OMP_OPTIONS_H

#include <cstddef>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** How the OpenMP backend shares the cells of a phase among its threads. */
        enum class OmpSchedule {
            /** A static `omp for` over the rows, the baseline. The default. */
            Rows,

            /** One contiguous range of cells in row-major order per thread. */
            Collapse2,

            /** A static `omp for` over tiles of tile_x by tile_y cells. */
            Tiles,

            /** One task per tile per phase, ordered by dependences on the neighbouring tiles. */
            Tasks
        };

        /** Where the OpenMP backend opens its parallel regions. */
        enum class OmpRegion {
            /** One region around the whole run, with the phases as work-sharing loops. */
            Run,

            /** One region per phase of every step. */
            Phase
        };

        /** Settings of the OpenMP backend, each with a usable default; the others ignore them. */
        struct OmpOptions
        {
            /** How the cells are shared among the threads. */
            OmpSchedule schedule = OmpSchedule::Rows;

            /** Thread count, 0 for the OpenMP default. */
            std::size_t threads = 0;

            /** Tile width in cells for Tiles and Tasks, positive. */
            std::size_t tile_x = 128;

            /** Tile height in cells for Tiles and Tasks, positive. */
            std::size_t tile_y = 32;

            /** Whether the inner loops carry `omp simd`. */
            bool simd = false;

            /** Where the parallel regions open; Tasks takes Run only. */
            OmpRegion region = OmpRegion::Run;
        };
    }
}

#endif
