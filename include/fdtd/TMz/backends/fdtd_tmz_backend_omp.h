/*
 * C++ header for the 2D TMz OpenMP backend.
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

#ifndef FDTD_TMZ_BACKEND_OMP_H
#define FDTD_TMZ_BACKEND_OMP_H

#include "fdtd/TMz/backends/fdtd_tmz_backend_omp_options.h"
#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_levels.h"
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

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
             * Multi-threaded CPU backend: the serial algorithm with its rows shared among threads.
             *
             * The level of the options picks the code that runs an advance, one file each in
             * `backends/omp/`; see the omp namespace. Every level matches serial within the
             * tolerance, and bit for bit in the exact build.
             */
            class OmpBackend final : public SerialBackend
            {
            public:
                /**
                 * Create the backend with zeroed fields.
                 *
                 * @param grid Grid the backend works on.
                 * @param options Settings of the backend.
                 */
                OmpBackend(const Grid &grid, const OmpOptions &options);

                /** The upload of fields, which the upload of coefficients would hide. */
                using SerialBackend::upload;

                /**
                 * Copy update coefficients into the backend, and from level v3 find the tuple the
                 * core cells share, with the PML of the coefficients in force and with none.
                 *
                 * @param coeffs Coefficients on the backend's grid.
                 */
                void upload(const Coeffs &coeffs) override;

                /**
                 * Short lowercase name of the backend.
                 *
                 * @return "openmp".
                 */
                std::string_view name() const noexcept override;

                /**
                 * Type of the backend.
                 *
                 * @return BackendType::OpenMP.
                 */
                BackendType type() const noexcept override;

                /**
                 * Settings of the backend.
                 *
                 * @return The options it was made with.
                 */
                const OmpOptions &options() const noexcept;

                /**
                 * Thread count the runs ask for: the option, or the OpenMP default for 0.
                 *
                 * @return The thread count.
                 */
                std::size_t threads() const noexcept;

                /**
                 * Modelled traffic of one step: one pass at v4, two below, and the core reading
                 * no arrays from v3 when it shares a tuple.
                 *
                 * @return two_pass_traffic() of the grid, the active PML, the boundary and the
                 *         model of the level.
                 */
                StepTraffic step_traffic() const noexcept override;

                /**
                 * Peaks of the CPU: the best bandwidth of the STREAM triad and its thread count,
                 * as machine_peaks() names them.
                 *
                 * @param bytes Size of each array of the triad.
                 * @param repeats Timed passes per thread count, at least one.
                 * @return `cpu_triad_bandwidth` in GB/s and `cpu_triad_threads`.
                 */
                static std::vector<MachinePeak> machine_peaks(std::size_t bytes,
                                                              std::size_t repeats);

            protected:
                /**
                 * Advance the fields and add the sources at the level of the options.
                 *
                 * @param nsteps Number of time steps.
                 * @param points Distinct cells inside the grid.
                 * @param values One value per point per step, step-major.
                 */
                void advance(std::size_t nsteps, std::span<const PointInjection> points,
                             std::span<const FloatType> values) override;

            private:
                /**
                 * Tuple the core cells share under the boundary in force, from level v3.
                 *
                 * @return The tuple and its core, or none when the core shares no tuple or the
                 *         level is below v3.
                 */
                const std::optional<omp::SharedTuple> &shared_tuple() const noexcept;

                /** Settings of the backend. */
                OmpOptions m_options;

                /** Thread count of the runs. */
                std::size_t m_threads;

                /** The rows of the grid, one part each. */
                serial::Partition m_partition;

                /** Tuple the core cells share with the PML of the coefficients in force. */
                std::optional<omp::SharedTuple> m_shared_pml;

                /** Tuple the core cells share with no PML in force. */
                std::optional<omp::SharedTuple> m_shared_plain;
            };

            /** Result of the STREAM triad. */
            struct TriadResult
            {
                /** Median bandwidth in bytes per second. */
                double bandwidth;

                /** Threads of the parallel regions. */
                std::size_t threads;
            };

            /**
             * Measure the memory bandwidth of the CPU with the STREAM triad, a = b + s c, on a
             * given number of threads.
             *
             * Each thread first touches the parts of the arrays it later updates. The bytes of a
             * pass are those of the three arrays, as STREAM counts them, so the read a store
             * makes before it writes is left out.
             *
             * @param bytes Size of each array, rounded down to a double.
             * @param repeats Timed passes after one untimed, at least one.
             * @param threads Threads of the parallel regions, 0 for the OpenMP default.
             * @return The median bandwidth and the thread count.
             */
            TriadResult triad_bandwidth(std::size_t bytes, std::size_t repeats,
                                        std::size_t threads);

            /**
             * Measure the triad on 1, 2, 4 and so on up to every thread OpenMP gives a parallel
             * region, and keep the fastest; on a machine shared with other processes, every
             * thread can be slower than fewer.
             *
             * @param bytes Size of each array, rounded down to a double.
             * @param repeats Timed passes per thread count after one untimed, at least one.
             * @return The best median bandwidth and its thread count.
             */
            TriadResult best_triad_bandwidth(std::size_t bytes, std::size_t repeats);
        }
    }
}

#endif
