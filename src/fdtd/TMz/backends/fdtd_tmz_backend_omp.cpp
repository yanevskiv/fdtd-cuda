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

// The table of distinct coefficients.
#include "fdtd/TMz/fdtd_tmz_coeff_table.h"

// Standard library for bounds, timing, sizes, views, errors and buffers.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
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
        // Check the options before anything is built from them.
        const Simulation::Options &checked(const Simulation::Options &options)
        {
            if (options.omp_fuse && options.omp_schedule != OmpSchedule::Rows) {
                throw std::invalid_argument("OmpBackend: fuse is for the Rows schedule only");
            }
            if (options.omp_fuse && options.omp_region == OmpRegion::Phase) {
                throw std::invalid_argument("OmpBackend: fuse runs in one region per run");
            }
            return options;
        }

        // Copy of a table entry, or none for a null one.
        std::optional<CoeffTuple> tuple_of(const CoeffTuple *entry)
        {
            if (entry == nullptr) {
                return std::nullopt;
            }
            return *entry;
        }

        // Cut the grid into the parts of the schedule: rows or one range per thread.
        cpu::Partition partition_for(const Grid &grid, const Simulation::Options &options,
                                     std::size_t threads)
        {
            if (options.omp_schedule == OmpSchedule::Collapse2) {
                return cpu::Partition::ranges(grid, threads);
            }
            return cpu::Partition::tiles(grid, grid.nx(), 1);
        }

        // Run a shared body on a team with the requested affinity policy.
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

        // Run every step as a static omp for over the parts, in one region or one per phase.
        template <bool Simd, bool Table, OmpRegion Region>
        void run_loops(const cpu::Stepper &stepper, std::size_t parts, std::size_t nsteps,
                       int threads, OmpBind bind)
        {
            if constexpr (Region == OmpRegion::Run) {
                run_team(threads, bind, [&] {
                    for (std::size_t n = 0; n < nsteps; ++n) {
#pragma omp for schedule(static)
                        for (std::size_t p = 0; p < parts; ++p) {
                            stepper.h_phase<Simd, Table>(p, n);
                        }
#pragma omp for schedule(static)
                        for (std::size_t p = 0; p < parts; ++p) {
                            stepper.e_phase<Simd, Table>(p, n);
                        }
                    }
                });
            } else {
                for (std::size_t n = 0; n < nsteps; ++n) {
                    run_team(threads, bind, [&] {
#pragma omp for schedule(static)
                        for (std::size_t p = 0; p < parts; ++p) {
                            stepper.h_phase<Simd, Table>(p, n);
                        }
                    });
                    run_team(threads, bind, [&] {
#pragma omp for schedule(static)
                        for (std::size_t p = 0; p < parts; ++p) {
                            stepper.e_phase<Simd, Table>(p, n);
                        }
                    });
                }
            }
        }

        // Sweep each thread's band of rows once a step, H then E row by row, after the top row's H.
        template <bool Simd, bool Table>
        void run_fused_rows(const cpu::Stepper &stepper, std::size_t rows, std::size_t nsteps,
                            int threads, OmpBind bind)
        {
            run_team(threads, bind, [&] {
                const auto team = static_cast<std::size_t>(omp_get_num_threads());
                const auto t = static_cast<std::size_t>(omp_get_thread_num());
                const std::size_t base = rows / team;
                const std::size_t extra = rows % team;
                const std::size_t begin = t * base + std::min(t, extra);
                const std::size_t end = begin + base + (t < extra ? 1 : 0);
                for (std::size_t n = 0; n < nsteps; ++n) {
                    if (begin < end) {
                        stepper.h_phase<Simd, Table>(end - 1, n);
                    }
#pragma omp barrier
                    for (std::size_t j = begin; j < end; ++j) {
                        if (j + 1 < end) {
                            stepper.h_phase<Simd, Table>(j, n);
                        }
                        stepper.e_phase<Simd, Table>(j, n);
                    }
#pragma omp barrier
                }
            });
        }

        // Hand the parts to the threads as the schedule, the fusion and the region say.
        template <bool Simd, bool Table>
        void run_schedule(const cpu::Stepper &stepper, const cpu::Partition &partition,
                          std::size_t nsteps, int threads, const Simulation::Options &options)
        {
            const std::size_t parts = partition.size();
            if (options.omp_fuse) {
                run_fused_rows<Simd, Table>(stepper, parts, nsteps, threads, options.omp_bind);
            } else if (options.omp_region == OmpRegion::Run) {
                run_loops<Simd, Table, OmpRegion::Run>(stepper, parts, nsteps, threads,
                                                       options.omp_bind);
            } else {
                run_loops<Simd, Table, OmpRegion::Phase>(stepper, parts, nsteps, threads,
                                                         options.omp_bind);
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

    // Copy the arrays in, then find the core's tuple with the coefficients' PML and with none.
    void OmpBackend::upload(const Coeffs &coeffs)
    {
        cpu::CpuBase::upload(coeffs);
        m_uniform_pml.reset();
        m_uniform_plain.reset();
        if (m_options.omp_table) {
            const CoeffTable table(coeffs);
            m_uniform_pml =
                tuple_of(table.uniform_entry(std::max<std::size_t>(coeffs.pml_thickness(), 1)));
            m_uniform_plain = tuple_of(table.uniform_entry(1));
        }
    }

    // Model one pass with fusion and two without, the core reading its tuple with the table.
    StepTraffic OmpBackend::step_traffic() const noexcept
    {
        TrafficModel model;
        model.one_pass = m_options.omp_fuse;
        model.table = uniform_tuple().has_value();
        return two_pass_traffic(grid(), active_pml(), boundary(), model);
    }

    // Pick the core's tuple for the boundary in force.
    const std::optional<CoeffTuple> &OmpBackend::uniform_tuple() const noexcept
    {
        return boundary() == Boundary::Pml ? m_uniform_pml : m_uniform_plain;
    }

    // View the arrays, with the core's tuple and the core when there is a tuple.
    cpu::CoeffsView OmpBackend::stepper_coeffs() const noexcept
    {
        cpu::CoeffsView view = coeffs();
        const std::optional<CoeffTuple> &uniform = uniform_tuple();
        if (uniform.has_value()) {
            const std::size_t lo = std::max<std::size_t>(active_pml(), 1);
            view.uniform = &*uniform;
            view.core = {
                .i0 = lo,
                .i1 = grid().nx() - lo - 1,
                .j0 = lo,
                .j1 = grid().ny() - lo - 1,
            };
        }
        return view;
    }

    // Build the stepper and hand its parts to the threads, with or without SIMD and the table.
    void OmpBackend::advance(std::size_t nsteps, std::span<const PointInjection> points,
                             std::span<const FloatType> values)
    {
        require_coeffs();
        FloatType *records = add_records(nsteps);
        const cpu::Stepper stepper(fields(), stepper_coeffs(), active_pml(), boundary(),
                                   m_partition, points, values, probes(), records);
        const int threads = static_cast<int>(m_threads);
        if (m_options.omp_simd && m_options.omp_table) {
            run_schedule<true, true>(stepper, m_partition, nsteps, threads, m_options);
        } else if (m_options.omp_simd) {
            run_schedule<true, false>(stepper, m_partition, nsteps, threads, m_options);
        } else if (m_options.omp_table) {
            run_schedule<false, true>(stepper, m_partition, nsteps, threads, m_options);
        } else {
            run_schedule<false, false>(stepper, m_partition, nsteps, threads, m_options);
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
