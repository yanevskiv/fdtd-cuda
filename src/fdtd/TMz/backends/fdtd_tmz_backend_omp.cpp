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
#include "fdtd/TMz/backends/fdtd_tmz_backend_omp.h"

// The stepper the threads share.
#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_stepper.h"

// OpenMP runtime for the default thread count.
#include <omp.h>

// Standard library for bounds, timing, sizes, views, errors and the dependence tokens.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Helpers private to this file.
    namespace
    {
        // Tasks made before the producer waits for them, so the tasks in flight stay bounded.
        constexpr std::size_t TASK_WINDOW = 4096;

        // Check the options before anything is built from them.
        const Simulation::Options &checked(const Simulation::Options &options)
        {
            if (options.omp_tile_x == 0 || options.omp_tile_y == 0) {
                throw std::invalid_argument("OmpBackend: tile sides must be positive");
            }
            if (options.omp_schedule == OmpSchedule::Tasks &&
                options.omp_region == OmpRegion::Phase) {
                throw std::invalid_argument("OmpBackend: Tasks run in one region per run");
            }
            return options;
        }

        // Cut the grid into the parts of the schedule: rows, one range per thread, or tiles.
        cpu::Partition partition_for(const Grid &grid, const Simulation::Options &options,
                                     std::size_t threads)
        {
            switch (options.omp_schedule) {
                case OmpSchedule::Rows:
                    return cpu::Partition::tiles(grid, grid.nx(), 1);
                case OmpSchedule::Collapse2:
                    return cpu::Partition::ranges(grid, threads);
                case OmpSchedule::Tiles:
                case OmpSchedule::Tasks:
                    break;
            }
            return cpu::Partition::tiles(grid, options.omp_tile_x, options.omp_tile_y);
        }

        // Run every step as a static omp for over the parts, in one region or one per phase.
        template <bool Simd, OmpRegion Region>
        void run_loops(const cpu::Stepper &stepper, std::size_t parts, std::size_t nsteps,
                       int threads)
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
        void run_loops(const cpu::Stepper &stepper, std::size_t parts, std::size_t nsteps,
                       int threads, OmpRegion region)
        {
            if (region == OmpRegion::Run) {
                run_loops<Simd, OmpRegion::Run>(stepper, parts, nsteps, threads);
            } else {
                run_loops<Simd, OmpRegion::Phase>(stepper, parts, nsteps, threads);
            }
        }

        // Make one task per tile per phase, each waiting for the phases of the tiles it reads.
        template <bool Simd>
        void run_tasks(const cpu::Stepper &stepper, const cpu::Partition &partition,
                       std::size_t nsteps, int threads)
        {
            const std::size_t parts = partition.size();
            const std::size_t columns = partition.tiles_x();
            const std::size_t rows = partition.tiles_y();
            const std::size_t window = std::max<std::size_t>(1, TASK_WINDOW / (2 * parts));
            std::vector<char> hTokens(parts);
            std::vector<char> eTokens(parts);
            char *h = hTokens.data();
            char *e = eTokens.data();
#pragma omp parallel num_threads(threads)
#pragma omp single
            for (std::size_t n = 0; n < nsteps; ++n) {
                for (std::size_t p = 0; p < parts; ++p) {
                    const std::size_t right = p % columns + 1 < columns ? p + 1 : p;
                    const std::size_t up = p / columns + 1 < rows ? p + columns : p;
#pragma omp task depend(in : e[p], e[right], e[up]) depend(out : h[p])
                    stepper.h_phase<Simd>(p, n);
                }
                for (std::size_t p = 0; p < parts; ++p) {
                    const std::size_t left = p % columns > 0 ? p - 1 : p;
                    const std::size_t down = p / columns > 0 ? p - columns : p;
#pragma omp task depend(in : h[p], h[left], h[down]) depend(out : e[p])
                    stepper.e_phase<Simd>(p, n);
                }
                if ((n + 1) % window == 0) {
#pragma omp taskwait
                }
            }
        }
    }

    // Check the options, fix the thread count and cut the grid for the schedule.
    OmpBackend::OmpBackend(const Grid &grid, const Simulation::Options &options)
        : cpu::CpuBase(grid)
        , m_options(checked(options))
        , m_threads(options.omp_threads > 0 ? options.omp_threads
                                            : static_cast<std::size_t>(omp_get_max_threads()))
        , m_partition(partition_for(grid, options, m_threads))
    {
        // Empty.
    }

    // Return the backend name.
    std::string_view OmpBackend::name() const noexcept
    {
        return backend_name(BackendType::OpenMP);
    }

    // Return the backend type.
    BackendType OmpBackend::type() const noexcept
    {
        return BackendType::OpenMP;
    }

    // Return the options.
    const Simulation::Options &OmpBackend::options() const noexcept
    {
        return m_options;
    }

    // Return the thread count.
    std::size_t OmpBackend::threads() const noexcept
    {
        return m_threads;
    }

    // Build the stepper and hand its parts to the threads as loops or tasks, with or without SIMD.
    void OmpBackend::advance(std::size_t nsteps, std::span<const PointInjection> points,
                             std::span<const FloatType> values)
    {
        require_coeffs();
        FloatType *records = add_records(nsteps);
        const cpu::Stepper stepper(fields(), coeffs(), active_pml(), boundary(), m_partition,
                                   points, values, probes(), records);
        const int threads = static_cast<int>(m_threads);
        const std::size_t parts = m_partition.size();
        const bool tasks = m_options.omp_schedule == OmpSchedule::Tasks;
        if (tasks && m_options.omp_simd) {
            run_tasks<true>(stepper, m_partition, nsteps, threads);
        } else if (tasks) {
            run_tasks<false>(stepper, m_partition, nsteps, threads);
        } else if (m_options.omp_simd) {
            run_loops<true>(stepper, parts, nsteps, threads, m_options.omp_region);
        } else {
            run_loops<false>(stepper, parts, nsteps, threads, m_options.omp_region);
        }
    }

    // Time each pass of the triad and take the median.
    TriadResult triad_bandwidth(std::size_t bytes, std::size_t repeats, std::size_t threads)
    {
        const int team = threads > 0 ? static_cast<int>(threads) : omp_get_max_threads();
        const std::size_t n = bytes / sizeof(double);
        const auto count = static_cast<std::ptrdiff_t>(n);
        const auto a = std::make_unique_for_overwrite<double[]>(n);
        const auto b = std::make_unique_for_overwrite<double[]>(n);
        const auto c = std::make_unique_for_overwrite<double[]>(n);
        double *pa = a.get();
        double *pb = b.get();
        double *pc = c.get();
#pragma omp parallel for schedule(static) num_threads(team)
        for (std::ptrdiff_t k = 0; k < count; ++k) {
            pa[k] = 0;
            pb[k] = 1;
            pc[k] = 2;
        }

        const double scalar = 3;
        std::vector<double> rates;
        for (std::size_t r = 0; r <= repeats; ++r) {
            const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(team)
            for (std::ptrdiff_t k = 0; k < count; ++k) {
                pa[k] = pb[k] + scalar * pc[k];
            }
            const auto stop = std::chrono::steady_clock::now();
            if (r > 0) {
                const double seconds = std::chrono::duration<double>(stop - start).count();
                rates.push_back(3.0 * static_cast<double>(n * sizeof(double)) / seconds);
            }
        }
        std::sort(rates.begin(), rates.end());
        return {
            .bandwidth = rates[rates.size() / 2],
            .threads = static_cast<std::size_t>(team),
        };
    }

    // Double the thread count up to the most, and keep the fastest.
    TriadResult best_triad_bandwidth(std::size_t bytes, std::size_t repeats)
    {
        const auto most = static_cast<std::size_t>(omp_get_max_threads());
        TriadResult best{.bandwidth = 0, .threads = 1};
        for (std::size_t threads = 1;; threads = std::min(2 * threads, most)) {
            const TriadResult triad = triad_bandwidth(bytes, repeats, threads);
            if (triad.bandwidth > best.bandwidth) {
                best = triad;
            }
            if (threads == most) {
                return best;
            }
        }
    }
}
