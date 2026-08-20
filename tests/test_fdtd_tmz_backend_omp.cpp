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
#include "fdtd/TMz/backends/fdtd_tmz_backend_omp.h"
#include "fdtd/TMz/backends/fdtd_tmz_backend_omp_options.h"
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_material.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"

// Standard library for math, limits, sizes, text, threads, errors and buffers.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Shorthand for the types under test.
using fdtd::FloatType;
using fdtd::TMz::BackendType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Coeffs;
using fdtd::TMz::Component;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;
using fdtd::TMz::Simulation;
using fdtd::TMz::backends::OmpBind;
using fdtd::TMz::backends::OmpOptions;
using fdtd::TMz::backends::OmpRegion;
using fdtd::TMz::backends::OmpSchedule;

// Helpers private to this file.
namespace
{
    // Steps of every run.
    constexpr std::size_t STEPS = 60;

    // Bound of the difference from serial, in machine epsilons per step, from the build.
    constexpr double TOLERANCE = FDTD_TEST_TOLERANCE;

    // Schedules built so far.
    constexpr OmpSchedule SCHEDULES[] = {OmpSchedule::Rows, OmpSchedule::Collapse2};

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
    Run run(BackendType backend, Simulation::Options opts, Size size, Boundary boundary,
            bool media = false)
    {
        opts.backend = backend;
        opts.boundary = boundary;
        opts.pml_thickness = 6;
        Simulation sim(size.nx, size.ny, opts);
        if (media) {
            fdtd::TMz::Material material = sim.material();
            material.fill_circle(double(size.nx) / 2, double(size.ny) / 2, 6,
                                 {.eps_r = 3, .sigma_e = FloatType(0.2)});
            material.fill_rect(8, 8, 14, size.ny - 8, {.eps_r = 2});
            sim.set_material(material);
        }
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

    // Run the same on the OpenMP backend, its settings given by their variant name.
    Run run(const OmpOptions &omp, Size size, Boundary boundary, bool media = false)
    {
        Simulation::Options opts;
        opts.variant = omp.name();
        opts.threads = omp.threads;
        return run(BackendType::OpenMP, opts, size, boundary, media);
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

    // Every combination of schedule, SIMD, table, fusion and region at a thread count.
    std::vector<OmpOptions> combinations(std::size_t threads)
    {
        std::vector<OmpOptions> all;
        for (const OmpSchedule schedule : SCHEDULES) {
            for (const bool simd : {false, true}) {
                for (const bool table : {false, true}) {
                    for (const bool fuse : {false, true}) {
                        for (const OmpRegion region : {OmpRegion::Run, OmpRegion::Phase}) {
                            const bool run = region == OmpRegion::Run;
                            if (fuse && (schedule != OmpSchedule::Rows || !run)) {
                                continue;
                            }
                            OmpOptions omp;
                            omp.schedule = schedule;
                            omp.simd = simd;
                            omp.table = table;
                            omp.fuse = fuse;
                            omp.region = region;
                            omp.threads = threads;
                            all.push_back(omp);
                        }
                    }
                }
            }
        }
        return all;
    }

    // Name the settings of a combination for the messages of a failed check.
    std::string describe(const OmpOptions &omp)
    {
        std::ostringstream text;
        text << "schedule ";
        text << int(omp.schedule);
        text << ", simd ";
        text << omp.simd;
        text << ", table ";
        text << omp.table;
        text << ", fuse ";
        text << omp.fuse;
        text << ", region ";
        text << int(omp.region);
        text << ", threads ";
        text << omp.threads;
        return text.str();
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
    const Run serial = run(BackendType::Serial, Simulation::Options{}, size, boundary);

    for (const std::size_t threads : thread_counts()) {
        for (const OmpOptions &omp : combinations(threads)) {
            INFO(describe(omp));
            const Run openmp = run(omp, size, boundary);
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
    const Run serial = run(BackendType::Serial, Simulation::Options{}, size, boundary);

    for (const std::size_t threads : thread_counts()) {
        for (const OmpOptions &omp : combinations(threads)) {
            INFO(describe(omp));
            const Run openmp = run(omp, size, boundary);
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
        INFO(describe(omp));
        const Run first = run(omp, size, Boundary::Pml);
        const Run second = run(omp, size, Boundary::Pml);
        CHECK(first.fields == second.fields);
        CHECK(first.probes == second.probes);
    }
}

// Verify that ranges and fused rows handle more threads than cells.
TEST_CASE("OpenMP handles more threads than cells", "[backend][openmp]")
{
    require_openmp();
    const Size size{9, 7};
    const Run serial = run(BackendType::Serial, Simulation::Options{}, size, Boundary::Pec);

    OmpOptions ranges;
    ranges.schedule = OmpSchedule::Collapse2;
    ranges.fuse = false;
    ranges.threads = 80;
    const Run many = run(ranges, size, Boundary::Pec);
    CHECK(relative_difference(serial.fields, many.fields) <= bound());
    CHECK(relative_difference(serial.probes, many.probes) <= bound());

    OmpOptions fused;
    fused.fuse = true;
    fused.threads = 12;
    const Run sweeps = run(fused, size, Boundary::Pec);
    CHECK(relative_difference(serial.fields, sweeps.fields) <= bound());
    CHECK(relative_difference(serial.probes, sweeps.probes) <= bound());
}

// Verify that the backend has its name and rejects options with no meaning.
TEST_CASE("OpenMP backend checks its options", "[unit][backend][openmp]")
{
    require_openmp();
    const fdtd::TMz::Grid grid(16, 16);
    Simulation::Options options;
    const auto backend = fdtd::TMz::make_backend(BackendType::OpenMP, grid, options);
    CHECK(backend->name() == "openmp");
    CHECK(backend->type() == BackendType::OpenMP);

    for (const char *variant : {"rows+fuse+phase", "collapse2+fuse"}) {
        CAPTURE(variant);
        options.variant = variant;
        CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::OpenMP, grid, options),
                        std::invalid_argument);
    }
    options.variant = "rows+fuse";
    CHECK_NOTHROW(fdtd::TMz::make_backend(BackendType::OpenMP, grid, options));
}

// Verify that the table and fusion match serial on media, where the core shares no tuple and
// reads the arrays, and in the exact build bit for bit.
TEST_CASE("OpenMP table and fusion match serial on media", "[backend][openmp]")
{
    require_openmp();
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Size size{67, 45};
    const Run serial = run(BackendType::Serial, Simulation::Options{}, size, boundary, true);
    for (const std::size_t threads : thread_counts()) {
        for (const bool fuse : {false, true}) {
            OmpOptions omp;
            omp.table = true;
            omp.fuse = fuse;
            omp.threads = threads;
            INFO(describe(omp));
            const Run openmp = run(omp, size, boundary, true);
            CHECK(relative_difference(serial.fields, openmp.fields) <= bound());
            CHECK(relative_difference(serial.probes, openmp.probes) <= bound());
            if (FDTD_EXACT) {
                CHECK(serial.fields == openmp.fields);
                CHECK(serial.probes == openmp.probes);
            }
        }
    }
}

// Verify the traffic model: the table takes 6 words a core cell on vacuum and none on media, and
// fusion is the one-pass model.
TEST_CASE("OpenMP models the traffic of the table and fusion", "[unit][backend][openmp]")
{
    require_openmp();
    const double word = sizeof(FloatType);
    for (const Boundary boundary : {Boundary::Pec, Boundary::Pmc, Boundary::Pml}) {
        INFO("boundary " << static_cast<int>(boundary));
        Simulation::Options options;
        options.backend = BackendType::OpenMP;
        options.boundary = boundary;
        options.pml_thickness = 2;
        options.variant = "rows+simd";
        const Simulation arrays(8, 6, options);
        options.variant = "rows+simd+table";
        const Simulation table(8, 6, options);
        const double core = boundary == Boundary::Pml ? 3 : 15;
        CHECK(table.step_traffic().bytes == arrays.step_traffic().bytes - 6 * core * word);
        CHECK(table.step_traffic().flops == arrays.step_traffic().flops);

        Simulation media(8, 6, options);
        fdtd::TMz::Material material = media.material();
        material.fill_rect(3, 2, 4, 3, {.eps_r = 2});
        media.set_material(material);
        CHECK(media.step_traffic().bytes == arrays.step_traffic().bytes);

        options.variant = "rows+simd+fuse";
        const Simulation fused(8, 6, options);
        fdtd::TMz::TrafficModel model;
        model.one_pass = true;
        const auto expected = fdtd::TMz::two_pass_traffic(
            fused.grid(), boundary == Boundary::Pml ? 2 : 0, boundary, model);
        CHECK(fused.step_traffic().bytes == expected.bytes);
        CHECK(fused.step_traffic().flops == expected.flops);
    }
}

// Verify that binding preserves fields and probes for every execution strategy.
TEST_CASE("OpenMP binding preserves results bit for bit", "[backend][openmp]")
{
    require_openmp();
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    for (const std::size_t threads : thread_counts()) {
        for (OmpOptions options : combinations(threads)) {
            const Run unbound = run(options, {67, 45}, boundary);
            for (const auto bind : {OmpBind::Close, OmpBind::Spread}) {
                options.bind = bind;
                INFO(describe(options));
                const Run bound = run(options, {67, 45}, boundary);
                CHECK(bound.fields == unbound.fields);
                CHECK(bound.probes == unbound.probes);
            }
        }
    }
}

#if FDTD_WITH_OPENMP

// Check that a tuple holds a cell's values exactly.
static void check_tuple(const fdtd::TMz::backends::serial::CoeffTuple &tuple, const Coeffs &coeffs,
                        std::size_t k)
{
    CHECK(tuple.ez_1 == coeffs.coeff_ez_x1()[k]);
    CHECK(tuple.ez_2 == coeffs.coeff_ez_x2()[k]);
    CHECK(tuple.ez_1 == coeffs.coeff_ez_y1()[k]);
    CHECK(tuple.ez_2 == coeffs.coeff_ez_y2()[k]);
    CHECK(tuple.hx_1 == coeffs.coeff_hx_1()[k]);
    CHECK(tuple.hx_2 == coeffs.coeff_hx_2()[k]);
    CHECK(tuple.hy_1 == coeffs.coeff_hy_1()[k]);
    CHECK(tuple.hy_2 == coeffs.coeff_hy_2()[k]);
}

// Verify that the core of vacuum shares one tuple until it is empty.
TEST_CASE("Core tuple of vacuum", "[unit][openmp][coeffs]")
{
    const Grid grid(9, 7, 1e-3F, 2e-3F);
    const Material material(grid);
    const Coeffs coeffs(material, courant_dt(grid, 0.9));
    const auto tuple = fdtd::TMz::backends::omp::core_tuple(coeffs, 1);
    REQUIRE(tuple.has_value());
    check_tuple(*tuple, coeffs, grid.index(4, 3));
    CHECK(fdtd::TMz::backends::omp::core_tuple(coeffs, 2).has_value());
    CHECK_FALSE(fdtd::TMz::backends::omp::core_tuple(coeffs, 3).has_value());
    CHECK_FALSE(fdtd::TMz::backends::omp::core_tuple(coeffs, 4).has_value());
}

// Verify that the core inside a PML shares one tuple, and a core reaching into the PML does not.
TEST_CASE("Core tuple of vacuum with a PML", "[unit][openmp][coeffs]")
{
    const Grid grid(30, 24);
    Material material(grid);
    material.set_pml(4, 3, 1e-6);
    const Coeffs coeffs(material, courant_dt(grid, 0.9));
    const auto tuple = fdtd::TMz::backends::omp::core_tuple(coeffs, 4);
    REQUIRE(tuple.has_value());
    check_tuple(*tuple, coeffs, grid.index(15, 12));
    CHECK_FALSE(fdtd::TMz::backends::omp::core_tuple(coeffs, 1).has_value());
}

// Verify that a core holding two media shares no tuple.
TEST_CASE("Core tuple of a slab and a lossy circle", "[unit][openmp][coeffs]")
{
    const Grid grid(40, 32);
    Material material(grid);
    material.fill_rect(10, 0, 16, 32, {.eps_r = 4});
    material.fill_circle(28, 16, 5, {.eps_r = 2, .sigma_e = FloatType(0.5)});
    const Coeffs coeffs(material, courant_dt(grid, 0.9));
    CHECK_FALSE(fdtd::TMz::backends::omp::core_tuple(coeffs, 1).has_value());
}

#endif
