/*
 * C++ file for the tests of the 2D TMz OpenMP backend against the serial one.
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

// Catch2 declarations for the tests.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

// Project headers under test.
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_omp_options.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"

// Standard library for math, limits, sizes, threads, errors and buffers.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

// Shorthand for the types under test.
using fdtd::FloatType;
using fdtd::TMz::BackendType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Component;
using fdtd::TMz::OmpOptions;
using fdtd::TMz::OmpRegion;
using fdtd::TMz::OmpSchedule;
using fdtd::TMz::Simulation;

// Helpers private to this file.
namespace
{
    // Steps of every run.
    constexpr std::size_t STEPS = 60;

    // Bound of the difference from serial, in machine epsilons per step, from the build.
    constexpr double TOLERANCE = FDTD_TEST_TOLERANCE;

    // Schedules built so far.
    constexpr OmpSchedule SCHEDULES[] = {OmpSchedule::Rows};

    // Result of a run: the fields and every probe's record.
    struct Run
    {
        /** Ez, Hx, Hy, Ezx and Ezy after the run, one after the other. */
        std::vector<FloatType> fields;

        /** Probe records, probe after probe. */
        std::vector<FloatType> probes;
    };

    // Grid of a run.
    struct Size
    {
        /** Cells along x. */
        std::size_t nx;

        /** Cells along y. */
        std::size_t ny;
    };

    // Append a span to a vector.
    void append(std::vector<FloatType> &out, std::span<const FloatType> values)
    {
        out.insert(out.end(), values.begin(), values.end());
    }

    // Run a pulse with an H source and probes on a backend, and collect the fields and records.
    Run run(BackendType backend, const OmpOptions &omp, Size size, Boundary boundary)
    {
        Simulation::Options opts;
        opts.backend = backend;
        opts.omp = omp;
        opts.boundary = boundary;
        opts.pml_thickness = 6;
        Simulation sim(size.nx, size.ny, opts);
        const auto pulse = fdtd::TMz::Gaussian::from_steps(sim.dt(), 8);
        sim.add_source(size.nx / 3, size.ny / 2, Component::Ez, pulse);
        sim.add_source(size.nx / 2 + 3, size.ny / 3, Component::Hy, pulse);
        sim.add_probe(size.nx / 2, size.ny / 2, Component::Ez);
        sim.add_probe(2, 3, Component::Ez);
        sim.add_probe(size.nx - 1, size.ny / 2, Component::Hx);
        sim.add_probe(size.nx / 2, size.ny - 2, Component::Hy);
        sim.run(STEPS / 2);
        sim.run(STEPS - STEPS / 2);
        sim.download();

        Run result;
        append(result.fields, sim.fields().ez());
        append(result.fields, sim.fields().hx());
        append(result.fields, sim.fields().hy());
        append(result.fields, sim.fields().ezx());
        append(result.fields, sim.fields().ezy());
        for (std::size_t k = 0; k < sim.probe_count(); ++k) {
            append(result.probes, sim.probe_data(k));
        }
        return result;
    }

    // Relative max-norm difference of two arrays, against the max-norm of the first.
    double relative_difference(std::span<const FloatType> ref, std::span<const FloatType> other)
    {
        double diff = 0;
        double norm = 0;
        for (std::size_t k = 0; k < ref.size(); ++k) {
            diff = std::max(diff, std::abs(double(ref[k]) - double(other[k])));
            norm = std::max(norm, std::abs(double(ref[k])));
        }
        return norm > 0 ? diff / norm : diff;
    }

    // Bound of the relative difference after STEPS steps.
    double bound()
    {
        return TOLERANCE * double(std::numeric_limits<FloatType>::epsilon()) * double(STEPS);
    }

    // Thread counts to try: 1, 2, 4 and the machine's.
    std::vector<std::size_t> thread_counts()
    {
        const std::size_t machine = std::max(1U, std::thread::hardware_concurrency());
        std::vector<std::size_t> counts{1, 2, 4};
        if (machine != 1 && machine != 2 && machine != 4) {
            counts.push_back(machine);
        }
        return counts;
    }

    // Every combination of schedule, SIMD and region at a thread count, as options.
    std::vector<OmpOptions> combinations(std::size_t threads)
    {
        std::vector<OmpOptions> all;
        for (const OmpSchedule schedule : SCHEDULES) {
            for (const bool simd : {false, true}) {
                for (const OmpRegion region : {OmpRegion::Run, OmpRegion::Phase}) {
                    if (schedule == OmpSchedule::Tasks && region == OmpRegion::Phase) {
                        continue;
                    }
                    OmpOptions omp;
                    omp.schedule = schedule;
                    omp.simd = simd;
                    omp.region = region;
                    omp.threads = threads;
                    omp.tile_x = 16;
                    omp.tile_y = 8;
                    all.push_back(omp);
                }
            }
        }
        return all;
    }

    // Skip the test when the OpenMP backend is not compiled in.
    void require_openmp()
    {
        if (!fdtd::TMz::is_backend_available(BackendType::OpenMP)) {
            SKIP("the OpenMP backend is not compiled in");
        }
    }
}

// Verify that every OpenMP combination matches serial within the tolerance.
TEST_CASE("OpenMP matches serial", "[backend][openmp]")
{
    require_openmp();
    const auto size = GENERATE(Size{64, 64}, Size{67, 45}, Size{45, 81});
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Run serial = run(BackendType::Serial, {}, size, boundary);

    for (const std::size_t threads : thread_counts()) {
        for (const OmpOptions &omp : combinations(threads)) {
            INFO("schedule " << int(omp.schedule) << ", simd " << omp.simd << ", region "
                             << int(omp.region) << ", threads " << threads);
            const Run openmp = run(BackendType::OpenMP, omp, size, boundary);
            CHECK(relative_difference(serial.fields, openmp.fields) <= bound());
            CHECK(relative_difference(serial.probes, openmp.probes) <= bound());
        }
    }
}

// Verify that in the exact build every OpenMP combination matches serial bit for bit.
TEST_CASE("OpenMP matches serial bit for bit", "[backend][openmp][exact]")
{
    require_openmp();
    if (!FDTD_EXACT) {
        SKIP("needs the exact build (FDTD_EXACT)");
    }
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Size size{67, 45};
    const Run serial = run(BackendType::Serial, {}, size, boundary);

    for (const std::size_t threads : thread_counts()) {
        for (const OmpOptions &omp : combinations(threads)) {
            INFO("schedule " << int(omp.schedule) << ", simd " << omp.simd << ", region "
                             << int(omp.region) << ", threads " << threads);
            const Run openmp = run(BackendType::OpenMP, omp, size, boundary);
            CHECK(serial.fields == openmp.fields);
            CHECK(serial.probes == openmp.probes);
        }
    }
}

// Verify that two runs of the same combination give the same bits, which catches data races.
TEST_CASE("OpenMP runs are deterministic", "[backend][openmp]")
{
    require_openmp();
    const Size size{67, 45};
    for (const OmpOptions &omp : combinations(thread_counts().back())) {
        INFO("schedule " << int(omp.schedule) << ", simd " << omp.simd << ", region "
                         << int(omp.region));
        const Run first = run(BackendType::OpenMP, omp, size, Boundary::Pml);
        const Run second = run(BackendType::OpenMP, omp, size, Boundary::Pml);
        CHECK(first.fields == second.fields);
        CHECK(first.probes == second.probes);
    }
}

// Verify that the backend has its name and rejects options with no meaning.
TEST_CASE("OpenMP backend checks its options", "[unit][backend][openmp]")
{
    require_openmp();
    const fdtd::TMz::Grid grid(16, 16);
    fdtd::TMz::BackendOptions options;
    const auto backend = fdtd::TMz::make_backend(BackendType::OpenMP, grid, options);
    CHECK(backend->name() == "openmp");
    CHECK(backend->type() == BackendType::OpenMP);

    options.omp.tile_x = 0;
    CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::OpenMP, grid, options),
                    std::invalid_argument);
    options.omp.tile_x = 8;
    options.omp.tile_y = 0;
    CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::OpenMP, grid, options),
                    std::invalid_argument);
}
