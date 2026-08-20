/*
 * C++ header for the 2D TMz OpenMP backend's thread team.
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

#ifndef FDTD_TMZ_BACKEND_OMP_TEAM_H
#define FDTD_TMZ_BACKEND_OMP_TEAM_H

#include "fdtd/TMz/backends/fdtd_tmz_backend_omp_options.h"

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /** Namespace for the levels of the 2D TMz OpenMP backend. */
            namespace omp
            {
                /**
                 * Run a body on every thread of one parallel region with an affinity policy.
                 *
                 * @tparam Body Callable taking nothing.
                 * @param threads Threads of the region.
                 * @param bind Affinity policy.
                 * @param body Body every thread runs.
                 */
                template <typename Body> void run_team(int threads, OmpBind bind, const Body &body)
                {
                    if (bind == OmpBind::Close) {
#pragma omp parallel num_threads(threads) proc_bind(close)
                        {
                            body();
                        }
                    } else if (bind == OmpBind::Spread) {
#pragma omp parallel num_threads(threads) proc_bind(spread)
                        {
                            body();
                        }
                    } else {
#pragma omp parallel num_threads(threads)
                        {
                            body();
                        }
                    }
                }
            }
        }
    }
}

#endif
