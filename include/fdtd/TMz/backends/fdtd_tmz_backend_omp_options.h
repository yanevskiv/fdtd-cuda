/*
 * C++ header for the settings of the 2D TMz OpenMP backend and the variant names they read from.
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

#ifndef FDTD_TMZ_BACKEND_OMP_OPTIONS_H
#define FDTD_TMZ_BACKEND_OMP_OPTIONS_H

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
            /** How the OpenMP backend shares the cells of a phase among its threads. */
            enum class OmpSchedule {
                /** A static `omp for` over the rows, the baseline. The default. */
                Rows,

                /** One contiguous range of cells in row-major order per thread. */
                Collapse2
            };

            /** Where the OpenMP backend opens its parallel regions. */
            enum class OmpRegion {
                /** One region around the whole run, with the phases as work-sharing loops. */
                Run,

                /** One region per phase of every step. */
                Phase
            };

            /** Affinity policy of the OpenMP backend's parallel regions. */
            enum class OmpBind {
                /** Use the runtime's policy. */
                Default,

                /** Keep the team's threads on nearby places. */
                Close,

                /** Spread the team's threads across the available places. */
                Spread
            };

            /**
             * Settings of the OpenMP backend, each with a usable default.
             *
             * A variant name, `<schedule>[+simd][+table][+fuse][+phase][+close|+spread]`, sets
             * every field but the thread count. The schedule is `rows` or `collapse2`; `+simd`,
             * `+table`, `+fuse` and `+phase` turn on SIMD, the coefficient table, row fusion and a
             * region per phase, each once, in any order. The backend, not the name, rejects the
             * combinations it cannot run, as `rows+fuse+phase`. `+close` and `+spread` select one
             * affinity policy; enable affinity in the runtime with OMP_PROC_BIND=true and
             * OMP_PLACES=threads. `default` names the default settings.
             */
            struct OmpOptions
            {
                /** How the cells are shared among the threads. */
                OmpSchedule schedule = OmpSchedule::Rows;

                /** Whether the inner loops carry `omp simd`; on by default, the fastest on the M5
                 * sweep. */
                bool simd = true;

                /** Where the parallel regions open. */
                OmpRegion region = OmpRegion::Run;

                /** Affinity policy; a runtime with affinity disabled ignores explicit policies,
                 * as required by OpenMP. */
                OmpBind bind = OmpBind::Default;

                /** Whether the core cells, the interior less its last column and row, read the
                 * one tuple of coefficients they share instead of the arrays, when they share
                 * one. */
                bool table = true;

                /** Whether each thread of the Rows schedule sweeps its band of rows once a step,
                 * H then E row by row, instead of an H pass and an E pass; Rows and Run only. */
                bool fuse = true;

                /** Thread count, 0 for the OpenMP default; no variant name sets it. */
                std::size_t threads = 0;

                /**
                 * Read the settings of a variant name.
                 *
                 * @param variant Variant name, as `rows+simd`, or `default`.
                 * @return The settings, with the default thread count.
                 * @throws std::invalid_argument When the name is malformed, with the reason and
                 *         the name in the message.
                 */
                static OmpOptions parse(std::string_view variant);

                /**
                 * Canonical variant name of the settings, which parse() reads back.
                 *
                 * The suffixes are `+simd`, `+table`, `+fuse`, `+phase`, then `+close` or
                 * `+spread` in that order; the thread count is left out.
                 *
                 * @return The name, as `rows+simd`.
                 */
                std::string name() const;
            };
        }
    }
}

#endif
