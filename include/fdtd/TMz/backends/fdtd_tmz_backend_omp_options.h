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
            /** Optimization level of the OpenMP backend, each the one before with one change. */
            enum class OmpLevel {
                /** Rows: the serial kernels, a static `omp for` over the rows. */
                V1,

                /** Simd: v1 with `omp simd` inner loops. */
                V2,

                /** Table: v2 with the core cells reading the one tuple of coefficients they
                 * share. */
                V3,

                /** Fused: v3 with H and E in one sweep of each thread's band of rows per step. The
                 * default. */
                V4
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
             * A variant name, `<level>[+close|+spread]`, sets every field but the thread count.
             * The level is `v1` to `v4`. `+close` and `+spread` select an affinity policy; enable
             * affinity in the runtime with OMP_PROC_BIND=true and OMP_PLACES=threads. `default`
             * names the default settings.
             */
            struct OmpOptions
            {
                /** Optimization level. */
                OmpLevel level = OmpLevel::V4;

                /** Affinity policy; a runtime with affinity disabled ignores explicit policies,
                 * as required by OpenMP. */
                OmpBind bind = OmpBind::Default;

                /** Thread count, 0 for the OpenMP default; no variant name sets it. */
                std::size_t threads = 0;

                /**
                 * Read the settings of a variant name.
                 *
                 * @param variant Variant name, as `v2+close`, or `default`.
                 * @return The settings, with the default thread count.
                 * @throws std::invalid_argument When the name is malformed, with the reason and
                 *         the name in the message.
                 */
                static OmpOptions parse(std::string_view variant);

                /**
                 * Canonical variant name of the settings, which parse() reads back.
                 *
                 * The level, then `+close` or `+spread`; the thread count is left out.
                 *
                 * @return The name, as `v2+close`.
                 */
                std::string name() const;
            };
        }
    }
}

#endif
