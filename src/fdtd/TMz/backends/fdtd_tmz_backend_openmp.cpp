/*
 * C++ file for the 2D TMz OpenMP backend.
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

// Class declaration.
#include "fdtd/TMz/backends/fdtd_tmz_backend_openmp.h"

// The stepper the threads share.
#include "fdtd/TMz/backends/fdtd_tmz_stepper.h"

// OpenMP runtime for the default thread count.
#include <omp.h>

// Standard library for sizes, views and errors.
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string_view>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Helpers private to this file.
    namespace
    {
        // Check the options before anything is built from them.
        const OmpOptions &checked(const OmpOptions &options)
        {
            if (options.tile_x == 0 || options.tile_y == 0) {
                throw std::invalid_argument("OpenMPBackend: tile sides must be positive");
            }
            if (options.schedule == OmpSchedule::Tasks) {
                throw std::invalid_argument("OpenMPBackend: the Tasks schedule is not built");
            }
            return options;
        }

        // Cut the grid into the parts of the schedule: rows, one range per thread, or tiles.
        Partition partition_for(const Grid &grid, const OmpOptions &options, std::size_t threads)
        {
            switch (options.schedule) {
                case OmpSchedule::Rows:
                    return Partition::tiles(grid, grid.nx(), 1);
                case OmpSchedule::Collapse2:
                    return Partition::ranges(grid, threads);
                case OmpSchedule::Tiles:
                case OmpSchedule::Tasks:
                    break;
            }
            return Partition::tiles(grid, options.tile_x, options.tile_y);
        }

        // Run every step as a static omp for over the parts per phase, in one region or one per
        // phase.
        template <bool Simd, OmpRegion Region>
        void run_loops(const Stepper &stepper, std::size_t parts, std::size_t nsteps, int threads)
        {
            if constexpr (Region == OmpRegion::Run) {
#pragma omp parallel num_threads(threads)
                for (std::size_t n = 0; n < nsteps; ++n) {
#pragma omp for schedule(static)
                    for (std::size_t p = 0; p < parts; ++p) {
                        stepper.h_phase<Simd>(p, n);
                    }
#pragma omp for schedule(static)
                    for (std::size_t p = 0; p < parts; ++p) {
                        stepper.e_phase<Simd>(p, n);
                    }
                }
            } else {
                for (std::size_t n = 0; n < nsteps; ++n) {
#pragma omp parallel for num_threads(threads) schedule(static)
                    for (std::size_t p = 0; p < parts; ++p) {
                        stepper.h_phase<Simd>(p, n);
                    }
#pragma omp parallel for num_threads(threads) schedule(static)
                    for (std::size_t p = 0; p < parts; ++p) {
                        stepper.e_phase<Simd>(p, n);
                    }
                }
            }
        }

        // Pick the region instance of the loops.
        template <bool Simd>
        void run_loops(const Stepper &stepper, std::size_t parts, std::size_t nsteps, int threads,
                       OmpRegion region)
        {
            if (region == OmpRegion::Run) {
                run_loops<Simd, OmpRegion::Run>(stepper, parts, nsteps, threads);
            } else {
                run_loops<Simd, OmpRegion::Phase>(stepper, parts, nsteps, threads);
            }
        }
    }

    // Check the options, fix the thread count and cut the grid for the schedule.
    OpenMPBackend::OpenMPBackend(const Grid &grid, const OmpOptions &options)
        : CpuBackend(grid)
        , m_options(checked(options))
        , m_threads(options.threads > 0 ? options.threads
                                        : static_cast<std::size_t>(omp_get_max_threads()))
        , m_partition(partition_for(grid, options, m_threads))
    {
        // Empty.
    }

    // Return the backend name.
    std::string_view OpenMPBackend::name() const noexcept
    {
        return backend_name(BackendType::OpenMP);
    }

    // Return the backend type.
    BackendType OpenMPBackend::type() const noexcept
    {
        return BackendType::OpenMP;
    }

    // Return the options.
    const OmpOptions &OpenMPBackend::options() const noexcept
    {
        return m_options;
    }

    // Return the thread count.
    std::size_t OpenMPBackend::threads() const noexcept
    {
        return m_threads;
    }

    // Build the stepper and hand its parts to the threads, with or without SIMD.
    void OpenMPBackend::advance(std::size_t nsteps, std::span<const PointInjection> points,
                                std::span<const FloatType> values)
    {
        require_coeffs();
        FloatType *records = add_records(nsteps);
        const Stepper stepper(fields(), coeffs(), active_pml(), boundary(), m_partition, points,
                              values, probes(), records);
        const int threads = static_cast<int>(m_threads);
        const std::size_t parts = m_partition.size();
        if (m_options.simd) {
            run_loops<true>(stepper, parts, nsteps, threads, m_options.region);
        } else {
            run_loops<false>(stepper, parts, nsteps, threads, m_options.region);
        }
    }
}
