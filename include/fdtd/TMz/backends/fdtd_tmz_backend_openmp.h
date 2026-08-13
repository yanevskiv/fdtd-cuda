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

#ifndef FDTD_TMZ_BACKEND_OPENMP_H
#define FDTD_TMZ_BACKEND_OPENMP_H

#include "fdtd/TMz/backends/fdtd_tmz_backend_cpu.h"
#include "fdtd/TMz/backends/fdtd_tmz_partition.h"
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_omp_options.h"

#include <cstddef>
#include <span>
#include <string_view>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /**
         * Multi-threaded CPU backend: the serial stencil, its parts shared among threads.
         *
         * The schedule of the options picks the partition and how its parts reach the
         * threads; the region and SIMD settings are compiled in as template instances and
         * picked when the run starts. The arithmetic per cell is the serial backend's.
         */
        class OpenMPBackend final : public CpuBackend
        {
        public:
            /**
             * Create the backend with zeroed fields.
             *
             * @param grid Grid the backend works on.
             * @param options Settings of the backend.
             * @throws std::invalid_argument When a tile side is zero, or the schedule is
             *         Tasks with a region per phase.
             */
            OpenMPBackend(const Grid &grid, const OmpOptions &options);

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
            OmpOptions m_options;

            /** Thread count of the runs. */
            std::size_t m_threads;

            /** Parts the threads share. */
            Partition m_partition;
        };
    }
}

#endif
