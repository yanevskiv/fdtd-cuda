/*
 * C++ file for the 2D TMz OpenMP backend's level v1, rows.
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

// Level declarations.
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_levels.h"

// The parallel region.
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_team.h"

// Standard library for sizes.
#include <cstddef>

// Namespace for the levels of the 2D TMz OpenMP backend.
namespace fdtd::TMz::backends::omp
{
    // Share the rows of each phase among the threads; the implied barrier ends each phase.
    void run_v1(const Advance &advance)
    {
        const serial::Stepper &stepper = advance.stepper;
        const std::size_t rows = stepper.partition().size();
        run_team(advance.threads, advance.bind, [&] {
            for (std::size_t n = 0; n < advance.nsteps; ++n) {
#pragma omp for schedule(static)
                for (std::size_t j = 0; j < rows; ++j) {
                    stepper.h_phase(j, n);
                }
#pragma omp for schedule(static)
                for (std::size_t j = 0; j < rows; ++j) {
                    stepper.e_phase(j, n);
                }
            }
        });
    }
}
