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

#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_base.h"
#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_partition.h"
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"

#include <cstddef>
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
            /**
             * Multi-threaded CPU backend: the serial stencil, its parts shared among threads.
             *
             * The schedule of the options picks the partition and how its parts reach the
             * threads: static loops over rows, ranges or tiles, or tasks over tiles that wait for
             * their neighbours instead of a barrier. The region and SIMD settings are compiled in
             * as template instances and picked when the run starts. The arithmetic per cell is the
             * serial backend's.
             */
            class OmpBackend final : public cpu::CpuBase
            {
            public:
                /**
                 * Create the backend with zeroed fields.
                 *
                 * @param grid Grid the backend works on.
                 * @param options Settings of a simulation; the backend reads the omp_ fields.
                 * @throws std::invalid_argument When a tile side is zero, or the schedule is
                 *         Tasks with a region per phase.
                 */
                OmpBackend(const Grid &grid, const Simulation::Options &options);

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
                const Simulation::Options &options() const noexcept;

                /**
                 * Thread count the runs ask for: the option, or the OpenMP default for 0.
                 *
                 * @return The thread count.
                 */
                std::size_t threads() const noexcept;

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
                /** Settings of the backend. */
                Simulation::Options m_options;

                /** Thread count of the runs. */
                std::size_t m_threads;

                /** Parts the threads share. */
                cpu::Partition m_partition;
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
