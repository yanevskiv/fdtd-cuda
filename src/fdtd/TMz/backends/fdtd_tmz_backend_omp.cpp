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

// The serial stepper the levels share.
#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"

// OpenMP runtime for the default thread count.
#include <omp.h>

// Standard library for bounds, timing, sizes, views and buffers.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Helpers private to this file.
    namespace
    {
        // The tuple the core of an interior with lower bound lo shares, with the core.
        std::optional<omp::SharedTuple> find_shared(const Coeffs &coeffs, std::size_t lo)
        {
            const std::optional<omp::CoeffTuple> tuple = omp::core_tuple(coeffs, lo);
            if (!tuple.has_value()) {
                return std::nullopt;
            }
            const Grid &grid = coeffs.grid();
            return omp::SharedTuple{
                .tuple = *tuple,
                .core =
                    {
                        .i0 = lo,
                        .i1 = grid.nx() - lo - 1,
                        .j0 = lo,
                        .j1 = grid.ny() - lo - 1,
                    },
            };
        }
    }

    // Fix the thread count and cut the grid into rows.
    OmpBackend::OmpBackend(const Grid &grid, const OmpOptions &options)
        : SerialBackend(grid)
        , m_options(options)
        , m_threads(options.threads > 0 ? options.threads
                                        : static_cast<std::size_t>(omp_get_max_threads()))
        , m_partition(serial::Partition::tiles(grid, grid.nx(), 1))
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
    const OmpOptions &OmpBackend::options() const noexcept
    {
        return m_options;
    }

    // Return the thread count.
    std::size_t OmpBackend::threads() const noexcept
    {
        return m_threads;
    }

    // Copy the arrays in, then from v3 find the core's tuple with the coefficients' PML and none.
    void OmpBackend::upload(const Coeffs &coeffs)
    {
        SerialBackend::upload(coeffs);
        m_shared_pml.reset();
        m_shared_plain.reset();
        if (m_options.level >= OmpLevel::V3) {
            m_shared_pml = find_shared(coeffs, std::max<std::size_t>(coeffs.pml_thickness(), 1));
            m_shared_plain = find_shared(coeffs, 1);
        }
    }

    // Model one pass at v4 and two below, the core reading its tuple when it shares one.
    StepTraffic OmpBackend::step_traffic() const noexcept
    {
        TrafficModel model;
        model.one_pass = m_options.level == OmpLevel::V4;
        model.table = shared_tuple().has_value();
        return two_pass_traffic(grid(), active_pml(), boundary(), model);
    }

    // Pick the core's tuple for the boundary in force.
    const std::optional<omp::SharedTuple> &OmpBackend::shared_tuple() const noexcept
    {
        return boundary() == Boundary::Pml ? m_shared_pml : m_shared_plain;
    }

    // Build the stepper over the rows and run the advance at the level of the options.
    void OmpBackend::advance(std::size_t nsteps, std::span<const PointInjection> points,
                             std::span<const FloatType> values)
    {
        require_coeffs();
        FloatType *records = add_records(nsteps);
        const serial::Stepper stepper(fields(), coeffs(), active_pml(), boundary(), m_partition,
                                      points, values, probes(), records);
        const omp::Advance run{
            .stepper = stepper,
            .nsteps = nsteps,
            .threads = static_cast<int>(m_threads),
            .bind = m_options.bind,
        };
        switch (m_options.level) {
            case OmpLevel::V1:
                omp::run_v1(run);
                return;
            case OmpLevel::V2:
                omp::run_v2(run);
                return;
            case OmpLevel::V3:
                omp::run_v3(run, shared_tuple());
                return;
            case OmpLevel::V4:
                break;
        }
        omp::run_v4(run, shared_tuple());
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
