/*
 * C++ file for the 2D TMz OpenMP backend's level v4, fused.
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

// The parallel region and v3's kernels.
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_team.h"
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_v3.h"

// OpenMP runtime for the team size and the thread number.
#include <omp.h>

// Standard library for bounds, sizes and the optional tuple.
#include <algorithm>
#include <cstddef>
#include <optional>

// Namespace for the levels of the 2D TMz OpenMP backend.
namespace fdtd::TMz::backends::omp
{
    // Give each thread a band of rows. Row j's Ez reads the H of rows j - 1 and j, and the next
    // band's first Ez reads this band's last H, so each thread runs that H first, before a
    // barrier, while every Ez it reads is still the old one. Then it runs each row's H and Ez in
    // turn, the row below's H done by then. A barrier ends each step.
    void run_v4(const Advance &advance, const std::optional<SharedTuple> &shared)
    {
        const serial::Stepper &stepper = advance.stepper;
        const std::size_t rows = stepper.partition().size();
        const SharedTuple *tuple = shared.has_value() ? &*shared : nullptr;
        run_team(advance.threads, advance.bind, [&] {
            const auto team = static_cast<std::size_t>(omp_get_num_threads());
            const auto t = static_cast<std::size_t>(omp_get_thread_num());
            const std::size_t base = rows / team;
            const std::size_t extra = rows % team;
            const std::size_t begin = t * base + std::min(t, extra);
            const std::size_t end = begin + base + (t < extra ? 1 : 0);
            for (std::size_t n = 0; n < advance.nsteps; ++n) {
                if (begin < end) {
                    v3::h_phase(stepper, tuple, end - 1, n);
                }

                #pragma omp barrier

                for (std::size_t j = begin; j < end; ++j) {
                    if (j + 1 < end) {
                        v3::h_phase(stepper, tuple, j, n);
                    }
                    v3::e_phase(stepper, tuple, j, n);
                }

                #pragma omp barrier
            }
        });
    }
}
