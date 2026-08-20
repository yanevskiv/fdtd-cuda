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
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"

#include <cstddef>
#include <optional>
#include <span>
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
            /** Namespace for what the 2D TMz OpenMP backend keeps on its own. */
            namespace omp
            {
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
                std::optional<serial::CoeffTuple> core_tuple(const Coeffs &coeffs, std::size_t lo);
            }

            /**
             * Multi-threaded CPU backend: the serial stencil, its parts shared among threads.
             *
             * The schedule of the options picks the partition and how its parts reach the
             * threads: static loops over rows or ranges, or, with fuse, one sweep of
             * each thread's band of rows per step. The region, SIMD and table settings are compiled
             * in as template instances and picked when the run starts. The arithmetic per cell is
             * the serial backend's.
             */
            class OmpBackend final : public SerialBackend
            {
            public:
                /**
                 * Create the backend with zeroed fields.
                 *
                 * @param grid Grid the backend works on.
                 * @param options Settings of the backend.
                 * @throws std::invalid_argument When fuse is set with a
                 * schedule other than Rows or a region per phase.
                 */
                OmpBackend(const Grid &grid, const OmpOptions &options);

                /** The upload of fields, which the upload of coefficients would hide. */
                using SerialBackend::upload;

                /**
                 * Copy update coefficients into the backend, and with table find the tuple
                 * the core cells share, with the PML of the coefficients in force and with none.
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
                 * Modelled traffic of one step: one pass with fuse, two otherwise, and the
                 * core reading no arrays when table finds a tuple it shares.
                 *
                 * @return two_pass_traffic() of the grid, the active PML, the boundary and the
                 *         model of the options.
                 */
                StepTraffic step_traffic() const noexcept override;

            protected:
                /**
                 * Advance the fields and add the sources, part by part on the threads.
                 *
                 * @param nsteps Number of time steps.
                 * @param points Distinct cells inside the grid.
                 * @param values One value per point per step, step-major.
                 */
                void advance(std::size_t nsteps, std::span<const PointInjection> points,
                             std::span<const FloatType> values) override;

            private:
                /**
                 * Tuple the core cells share under the boundary in force, with table.
                 *
                 * @return The tuple, or empty when the core does not share one or the option is
                 *         off.
                 */
                const std::optional<serial::CoeffTuple> &uniform_tuple() const noexcept;

                /**
                 * View of the coefficient arrays, with the uniform tuple and its core when there
                 * is one.
                 *
                 * @return The view the stepper reads.
                 */
                serial::CoeffsView stepper_coeffs() const noexcept;

                /** Settings of the backend. */
                OmpOptions m_options;

                /** Thread count of the runs. */
                std::size_t m_threads;

                /** Parts the threads share. */
                serial::Partition m_partition;

                /** Tuple the core cells share with the PML of the coefficients in force. */
                std::optional<serial::CoeffTuple> m_uniform_pml;

                /** Tuple the core cells share with no PML in force. */
                std::optional<serial::CoeffTuple> m_uniform_plain;
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
