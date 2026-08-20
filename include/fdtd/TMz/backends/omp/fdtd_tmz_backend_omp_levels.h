/*
 * C++ header for the 2D TMz OpenMP backend's levels.
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

#ifndef FDTD_TMZ_BACKEND_OMP_LEVELS_H
#define FDTD_TMZ_BACKEND_OMP_LEVELS_H

#include "fdtd/TMz/backends/fdtd_tmz_backend_omp_options.h"
#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <optional>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /**
             * Namespace for the levels of the 2D TMz OpenMP backend, one file each.
             *
             * Each level is the one before it with one optimization:
             *
             * - v1, rows: the serial kernels, a static `omp for` over the rows in one parallel
             *   region per advance.
             * - v2, simd: v1 with kernels of its own whose inner loops carry `omp simd`.
             * - v3, table: v2 with the core cells reading the one tuple of coefficients they
             *   share instead of six arrays.
             * - v4, fused: v3 with H and E in one sweep of each thread's band of rows per step.
             *
             * The headers of the kernels hold OpenMP pragmas and are included by the levels'
             * sources alone; this one holds none.
             */
            namespace omp
            {
                /**
                 * The update coefficients of a cell whose Ez takes the same pair along x and y, as
                 * every cell outside the PML does: the pairs of Ez, Hx and Hy.
                 */
                struct CoeffTuple
                {
                    /** Ez decay factor, along x and y alike. */
                    FloatType ez_1;

                    /** Ez curl factor, along x and y alike, before 1/dx or 1/dy. */
                    FloatType ez_2;

                    /** Hx decay factor. */
                    FloatType hx_1;

                    /** Hx curl factor, before 1/dy. */
                    FloatType hx_2;

                    /** Hy decay factor. */
                    FloatType hy_1;

                    /** Hy curl factor, before 1/dx. */
                    FloatType hy_2;
                };

                /** The tuple every cell of the core holds in the arrays, and the core. */
                struct SharedTuple
                {
                    /** The tuple. */
                    CoeffTuple tuple;

                    /** The core of the interior: the interior less its last column and row,
                     * where Hy and Hx sit half a cell into the PML. */
                    serial::Block core;
                };

                /**
                 * The one tuple every cell of the core of an interior holds, compared bit by bit.
                 *
                 * The core is the interior less its last column and row, where Hy and Hx sit half a
                 * cell into the PML and take its grading.
                 *
                 * @param coeffs Coefficients to read.
                 * @param lo Lower bound of the interior along both axes: the core holds the cells
                 *        with lo <= i < nx - lo - 1 and lo <= j < ny - lo - 1.
                 * @return The tuple, or none when the core is empty, or two of its cells differ, or
                 *         a cell's Ez pairs along x and y differ.
                 */
                std::optional<CoeffTuple> core_tuple(const Coeffs &coeffs, std::size_t lo);

                /** What every level needs to run an advance. */
                struct Advance
                {
                    /** Stepper over the rows of the grid, one part per row. */
                    const serial::Stepper &stepper;

                    /** Number of time steps. */
                    std::size_t nsteps;

                    /** Threads of the parallel region. */
                    int threads;

                    /** Affinity policy of the parallel region. */
                    OmpBind bind;
                };

                /**
                 * Run an advance at level v1: each step an `omp for` over the rows for the H
                 * phase, then one for the E phase, with the serial kernels.
                 *
                 * @param advance The advance.
                 */
                void run_v1(const Advance &advance);

                /**
                 * Run an advance at level v2: v1 with `omp simd` inner loops.
                 *
                 * @param advance The advance.
                 */
                void run_v2(const Advance &advance);

                /**
                 * Run an advance at level v3: v2 with the core cells reading the shared tuple.
                 *
                 * @param advance The advance.
                 * @param shared The tuple the core shares, or none to read the arrays everywhere.
                 */
                void run_v3(const Advance &advance, const std::optional<SharedTuple> &shared);

                /**
                 * Run an advance at level v4: v3 with each thread sweeping its band of rows once a
                 * step, H then E row by row, after the H of the band's last row.
                 *
                 * @param advance The advance.
                 * @param shared The tuple the core shares, or none to read the arrays everywhere.
                 */
                void run_v4(const Advance &advance, const std::optional<SharedTuple> &shared);
            }
        }
    }
}

#endif
