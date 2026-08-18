/*
 * C++ file for the tests of the 2D TMz CUDA backend against the serial one.
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
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_material.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"

// Standard library for algorithms, math, limits, sizes, errors and buffers.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

// Shorthand for the types under test.
using fdtd::FloatType;
using fdtd::TMz::BackendType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Coeffs;
using fdtd::TMz::Component;
using fdtd::TMz::CudaKernel;
using fdtd::TMz::Fields;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;
using fdtd::TMz::PointInjection;
using fdtd::TMz::PointProbe;
using fdtd::TMz::Simulation;

// Helpers private to this file.
namespace
{
    // Steps of every run.
    constexpr std::size_t STEPS = 60;

    // Bound of the difference from serial, in machine epsilons per step, from the build.
    constexpr double TOLERANCE = FDTD_TEST_CUDA_TOLERANCE;

    // Kernel layouts built so far.
    constexpr CudaKernel KERNELS[] = {CudaKernel::Fused, CudaKernel::Split, CudaKernel::Tiled};

    // Result of a run: the fields and every probe's record.
    struct Run
    {
        /** Ez, Hx, Hy, Ezx and Ezy after the run, one after the other. */
        std::vector<FloatType> fields;

        /** Probe records, as fetched. */
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

    // Thread block of a run.
    struct Block
    {
        /** Threads along x. */
        std::size_t x;

        /** Threads along y. */
        std::size_t y;
    };

    // Append a span to a vector.
    void append(std::vector<FloatType> &out, std::span<const FloatType> values)
    {
        out.insert(out.end(), values.begin(), values.end());
    }

    // Append the five fields to a vector.
    void append_fields(std::vector<FloatType> &out, const Fields &fields)
    {
        append(out, fields.ez());
        append(out, fields.hx());
        append(out, fields.hy());
        append(out, fields.ezx());
        append(out, fields.ezy());
    }

    // Fill every field with distinct values that keep a few steps bounded.
    void fill_distinct(Fields &fields)
    {
        for (std::size_t k = 0; k < fields.grid().size(); ++k) {
            const double x = static_cast<double>(k % 17) - 8;
            fields.ez()[k] = FloatType(x / 8);
            fields.hx()[k] = FloatType(x / 16);
            fields.hy()[k] = FloatType(-x / 32);
            fields.ezx()[k] = FloatType(x / 4);
            fields.ezy()[k] = FloatType(-x / 4);
        }
    }

    // Options of a CUDA run with a kernel and a block.
    Simulation::Options cuda_options(CudaKernel kernel, Block block)
    {
        Simulation::Options options;
        options.cuda_kernel = kernel;
        options.cuda_block_x = block.x;
        options.cuda_block_y = block.y;
        return options;
    }

    // Run a pulse with an H source and probes through a Simulation, and collect the results.
    Run run(BackendType backend, Simulation::Options opts, Size size, Boundary boundary)
    {
        opts.backend = backend;
        opts.boundary = boundary;
        opts.pml_thickness = 6;
        Simulation sim(size.nx, size.ny, opts);
        const auto pulse = fdtd::TMz::Gaussian::from_steps(sim.dt(), 8);
        sim.add_source(size.nx / 3, size.ny / 2, Component::Ez, pulse);
        sim.add_source(size.nx / 2 + 3, size.ny / 3, Component::Hy, pulse);
        sim.add_source(size.nx / 2 - 2, size.ny / 2 + 3, Component::Hx, pulse);
        sim.add_probe(size.nx / 2, size.ny / 2, Component::Ez);
        sim.add_probe(2, 3, Component::Ez);
        sim.add_probe(size.nx - 1, size.ny / 2, Component::Hx);
        sim.add_probe(size.nx / 2, size.ny - 2, Component::Hy);
        sim.run(STEPS / 2);
        sim.run(STEPS - STEPS / 2);
        sim.download();

        Run result;
        append_fields(result.fields, sim.fields());
        for (std::size_t k = 0; k < sim.probe_count(); ++k) {
            append(result.probes, sim.probe_data(k));
        }
        return result;
    }

    // Step a backend from distinct fields, with sources and probes on any cells, and collect.
    Run run_backend(BackendType type, const Simulation::Options &options, const Grid &grid,
                    Boundary boundary, std::size_t pml)
    {
        Material material(grid);
        material.set_pml(pml, 3, 1e-6);
        const Coeffs coeffs(material, fdtd::TMz::courant_dt(grid, 0.9));
        Fields fields(grid);
        fill_distinct(fields);
        const auto backend = fdtd::TMz::make_backend(type, grid, options);
        backend->upload(fields);
        backend->upload(coeffs);
        backend->set_boundary(boundary);

        const std::size_t last = grid.index(grid.nx() - 1, grid.ny() - 1);
        const std::vector<PointProbe> probes{{Component::Ez, 0},
                                             {Component::Hx, last},
                                             {Component::Hy, last / 2},
                                             {Component::Ez, last / 2}};
        backend->set_probes(probes);
        const std::vector<PointInjection> points{
            {Component::Ez, last / 2}, {Component::Hx, 0}, {Component::Hy, last}};
        std::vector<FloatType> values(STEPS * points.size());
        for (std::size_t k = 0; k < values.size(); ++k) {
            values[k] = FloatType(k % 5) / FloatType(10);
        }
        backend->step(STEPS / 3);
        backend->step(STEPS - STEPS / 3, std::span(points),
                      std::span(values).first((STEPS - STEPS / 3) * points.size()));
        backend->download(fields);

        Run result;
        append_fields(result.fields, fields);
        result.probes = backend->fetch_probes();
        return result;
    }

    // Step a backend through advances of the given lengths, each with sources on Ez, Hx and Hy
    // and values of its own, and probes set before the first and again before the last.
    Run run_advances(const Simulation::Options &options, const Grid &grid, Boundary boundary,
                     std::span<const std::size_t> lengths)
    {
        Material material(grid);
        material.set_pml(5, 3, 1e-6);
        const Coeffs coeffs(material, fdtd::TMz::courant_dt(grid, 0.9));
        Fields fields(grid);
        fill_distinct(fields);
        const auto backend = fdtd::TMz::make_backend(BackendType::Cuda, grid, options);
        backend->upload(fields);
        backend->upload(coeffs);
        backend->set_boundary(boundary);

        const std::size_t last = grid.index(grid.nx() - 1, grid.ny() - 1);
        const std::vector<PointProbe> probes{{Component::Ez, last / 2}, {Component::Hy, last}};
        const std::vector<PointInjection> points{
            {Component::Hx, last / 3}, {Component::Ez, last / 2}, {Component::Hy, 1}};
        Run result;
        backend->set_probes(probes);
        for (std::size_t a = 0; a < lengths.size(); ++a) {
            if (a + 1 == lengths.size()) {
                append(result.probes, backend->fetch_probes());
                backend->set_probes(std::span(probes).first(1));
            }
            std::vector<FloatType> values(lengths[a] * points.size());
            for (std::size_t k = 0; k < values.size(); ++k) {
                values[k] = FloatType((k + a) % 7) / FloatType(10);
            }
            backend->step(lengths[a], std::span(points), std::span(values));
        }
        backend->download(fields);
        append_fields(result.fields, fields);
        append(result.probes, backend->fetch_probes());
        return result;
    }

    // Step a backend from distinct fields on a material, with sources and probes, and collect.
    Run run_material(BackendType type, const Simulation::Options &options, const Material &material,
                     Boundary boundary)
    {
        const Grid &grid = material.grid();
        const Coeffs coeffs(material, fdtd::TMz::courant_dt(grid, 0.9));
        Fields fields(grid);
        fill_distinct(fields);
        const auto backend = fdtd::TMz::make_backend(type, grid, options);
        backend->upload(fields);
        backend->upload(coeffs);
        backend->set_boundary(boundary);
        const std::size_t last = grid.index(grid.nx() - 1, grid.ny() - 1);
        const std::vector<PointProbe> probes{{Component::Ez, last / 2}, {Component::Hx, 7}};
        backend->set_probes(probes);
        const std::vector<PointInjection> points{{Component::Ez, last / 3}, {Component::Hy, 9}};
        std::vector<FloatType> values(STEPS * points.size());
        for (std::size_t k = 0; k < values.size(); ++k) {
            values[k] = FloatType(k % 3) / FloatType(8);
        }
        backend->step(STEPS, std::span(points), std::span(values));
        backend->download(fields);
        Run result;
        append_fields(result.fields, fields);
        result.probes = backend->fetch_probes();
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

    // Skip the test when the CUDA backend is not compiled in or no device is present.
    void require_cuda()
    {
        if (!fdtd::TMz::is_backend_available(BackendType::Cuda)) {
            SKIP("the CUDA backend is not compiled in, or no device is present");
        }
    }
}

// Verify that every kernel and block size matches serial within the tolerance.
TEST_CASE("CUDA matches serial", "[backend][cuda]")
{
    require_cuda();
    const auto size = GENERATE(Size{64, 64}, Size{67, 45}, Size{45, 81});
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Run serial = run(BackendType::Serial, {}, size, boundary);

    for (const CudaKernel kernel : KERNELS) {
        for (const Block block : {Block{32, 8}, Block{16, 16}, Block{128, 1}, Block{7, 5}}) {
            INFO("size " << size.nx << "x" << size.ny << ", boundary " << int(boundary)
                         << ", kernel " << int(kernel) << ", block " << block.x << "x" << block.y);
            const Run cuda = run(BackendType::Cuda, cuda_options(kernel, block), size, boundary);
            CHECK(relative_difference(serial.fields, cuda.fields) <= bound());
            CHECK(relative_difference(serial.probes, cuda.probes) <= bound());
        }
    }
}

// Verify grids smaller than a block and sources and probes on any cell, the edge included.
TEST_CASE("CUDA matches serial on small grids", "[backend][cuda]")
{
    require_cuda();
    const auto size = GENERATE(Size{1, 1}, Size{2, 5}, Size{9, 7}, Size{33, 2}, Size{31, 40});
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Grid grid(size.nx, size.ny, FloatType(1e-3), FloatType(2e-3));
    const std::size_t pml = boundary == Boundary::Pml && size.nx > 8 && size.ny > 6 ? 3 : 0;
    const Run serial = run_backend(BackendType::Serial, {}, grid, boundary, pml);

    for (const CudaKernel kernel : KERNELS) {
        for (const Block block : {Block{32, 8}, Block{7, 5}, Block{1, 1}}) {
            INFO("size " << size.nx << "x" << size.ny << ", boundary " << int(boundary)
                         << ", kernel " << int(kernel) << ", block " << block.x << "x" << block.y);
            const Run cuda =
                run_backend(BackendType::Cuda, cuda_options(kernel, block), grid, boundary, pml);
            CHECK(relative_difference(serial.fields, cuda.fields) <= bound());
            CHECK(relative_difference(serial.probes, cuda.probes) <= bound());
        }
    }
}

// Verify that in the exact build every kernel matches serial bit for bit.
TEST_CASE("CUDA matches serial bit for bit", "[backend][cuda][exact]")
{
    require_cuda();
    if (!FDTD_EXACT) {
        SKIP("needs the exact build (FDTD_EXACT)");
    }
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Size size{67, 45};
    const Run serial = run(BackendType::Serial, {}, size, boundary);
    const Run small = run_backend(BackendType::Serial, {}, Grid(9, 7), boundary, 0);

    for (const CudaKernel kernel : KERNELS) {
        for (const Block block : {Block{32, 8}, Block{7, 5}}) {
            INFO("kernel " << int(kernel) << ", block " << block.x << "x" << block.y);
            const Simulation::Options options = cuda_options(kernel, block);
            const Run cuda = run(BackendType::Cuda, options, size, boundary);
            CHECK(serial.fields == cuda.fields);
            CHECK(serial.probes == cuda.probes);
            const Run tiny = run_backend(BackendType::Cuda, options, Grid(9, 7), boundary, 0);
            CHECK(small.fields == tiny.fields);
            CHECK(small.probes == tiny.probes);
        }
    }
}

// Verify that two runs of the same kernel give the same bits.
TEST_CASE("CUDA runs are deterministic", "[backend][cuda]")
{
    require_cuda();
    for (const CudaKernel kernel : KERNELS) {
        INFO("kernel " << int(kernel));
        const Simulation::Options options = cuda_options(kernel, {32, 8});
        const Run first = run(BackendType::Cuda, options, {67, 45}, Boundary::Pml);
        const Run second = run(BackendType::Cuda, options, {67, 45}, Boundary::Pml);
        CHECK(first.fields == second.fields);
        CHECK(first.probes == second.probes);
    }
}

// Verify that a run split in two gives the bits of one run, snapshots and probes included.
TEST_CASE("CUDA runs split and snapshots", "[backend][cuda]")
{
    require_cuda();
    Simulation::Options options;
    options.backend = BackendType::Cuda;
    options.boundary = Boundary::Pml;
    options.pml_thickness = 6;
    Simulation whole(50, 40, options);
    Simulation split(50, 40, options);
    options.backend = BackendType::Serial;
    Simulation serial(50, 40, options);
    for (Simulation *sim : {&whole, &split, &serial}) {
        sim->add_source(20, 20, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim->dt(), 8));
        sim->add_probe(25, 22, Component::Ez);
        sim->add_snapshot(17);
        sim->add_snapshot(33);
    }
    whole.run(STEPS);
    split.run(STEPS / 4);
    split.run(STEPS - STEPS / 4);
    serial.run(STEPS);
    whole.download();
    split.download();
    serial.download();

    std::vector<FloatType> a;
    std::vector<FloatType> b;
    std::vector<FloatType> c;
    append_fields(a, whole.fields());
    append_fields(b, split.fields());
    append_fields(c, serial.fields());
    CHECK(a == b);
    CHECK(whole.probe_data(0) == split.probe_data(0));
    CHECK(relative_difference(c, a) <= bound());
    CHECK(relative_difference(serial.probe_data(0), whole.probe_data(0)) <= bound());

    REQUIRE(whole.snapshots().size() == 2);
    REQUIRE(serial.snapshots().size() == 2);
    for (std::size_t s = 0; s < 2; ++s) {
        CHECK(whole.snapshots()[s].step == serial.snapshots()[s].step);
        std::vector<FloatType> gpu;
        std::vector<FloatType> cpu;
        append_fields(gpu, whole.snapshots()[s].fields);
        append_fields(cpu, serial.snapshots()[s].fields);
        CHECK(relative_difference(cpu, gpu) <= bound());
    }
}

// Verify that fields come back as they went in, whatever the padding of the device rows.
TEST_CASE("CUDA upload and download round trip", "[backend][cuda]")
{
    require_cuda();
    const auto size = GENERATE(Size{1, 1}, Size{31, 3}, Size{33, 4}, Size{67, 45});
    const Grid grid(size.nx, size.ny);
    Fields in(grid);
    fill_distinct(in);
    const auto backend = fdtd::TMz::make_backend(BackendType::Cuda, grid);
    backend->upload(in);
    Fields out(grid);
    backend->download(out);

    std::vector<FloatType> a;
    std::vector<FloatType> b;
    append_fields(a, in);
    append_fields(b, out);
    CHECK(a == b);

    CHECK_THROWS_AS(backend->upload(Fields(Grid(size.nx + 1, size.ny))), std::invalid_argument);
    Fields other(Grid(size.nx, size.ny + 1));
    CHECK_THROWS_AS(backend->download(other), std::invalid_argument);
}

// Verify that the records grow over several advances and start again when the probes change.
TEST_CASE("CUDA probes over several advances", "[backend][cuda]")
{
    require_cuda();
    const Grid grid(20, 16);
    const auto backend = fdtd::TMz::make_backend(BackendType::Cuda, grid);
    backend->upload(Coeffs(Material(grid), fdtd::TMz::courant_dt(grid, 0.9)));
    const std::vector<PointProbe> probes{{Component::Ez, grid.index(10, 8)},
                                         {Component::Ez, grid.index(10, 8)}};
    backend->set_probes(probes);
    const std::vector<PointInjection> points{{Component::Ez, grid.index(10, 8)}};
    const std::vector<FloatType> values{FloatType(1), FloatType(2), FloatType(3)};
    backend->step(3, std::span(points), std::span(values));
    backend->step(2);
    CHECK(backend->probe_count() == 2);
    CHECK(backend->probe_steps() == 5);
    const std::vector<FloatType> records = backend->fetch_probes();
    REQUIRE(records.size() == 10);
    CHECK(records[0] == FloatType(1));
    CHECK(records[0] == records[1]);
    CHECK(records[8] == records[9]);

    backend->set_probes(std::span(probes).first(1));
    CHECK(backend->probe_steps() == 0);
    backend->step(4);
    CHECK(backend->probe_count() == 1);
    CHECK(backend->probe_steps() == 4);
    CHECK(backend->fetch_probes().size() == 4);
}

// Verify that replaying captured steps as a graph gives the plain launches' results bit for bit:
// advances shorter than a graph, as long as one and longer, the sources and probes changing
// between them so the graph is captured again, and the remainder steps launched plainly.
TEST_CASE("CUDA graphs match plain launches bit for bit", "[backend][cuda]")
{
    require_cuda();
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Grid grid(45, 37, FloatType(1e-3), FloatType(2e-3));
    const std::vector<std::size_t> lengths{5, 16, 37, 0, 33};
    for (const CudaKernel kernel : KERNELS) {
        INFO("kernel " << static_cast<int>(kernel));
        auto options = cuda_options(kernel, {32, 8});
        const Run plain = run_advances(options, grid, boundary, lengths);
        options.cuda_graph = true;
        const Run graph = run_advances(options, grid, boundary, lengths);
        CHECK(graph.fields == plain.fields);
        CHECK(graph.probes == plain.probes);
        CHECK(plain.probes.size() == 2 * (5 + 16 + 37) + 33);
    }
}

// Verify that reading the coefficients from the table gives the arrays' results bit for bit, on
// vacuum with a PML (one tuple for the interior), on a slab and a lossy circle (a byte per cell),
// and past the table's capacity (cells that read the arrays among cells that read the table).
TEST_CASE("CUDA coefficient table matches the arrays bit for bit", "[backend][cuda]")
{
    require_cuda();
    const Grid grid(46, 38, FloatType(1e-3), FloatType(2e-3));
    Material vacuum(grid);
    vacuum.set_pml(6, 3, 1e-6);
    Material media(grid);
    media.set_pml(5, 3, 1e-6);
    media.fill_rect(12, 0, 18, 38, {.eps_r = 4});
    media.fill_circle(30, 19, 6, {.eps_r = 2, .sigma_e = FloatType(0.5)});
    Material crowded(grid);
    for (std::size_t k = 0; k < 400; ++k) {
        const std::size_t i = 3 + k % 40;
        const std::size_t j = 3 + k / 40;
        crowded.fill_rect(i, j, i + 1, j + 1, {.eps_r = FloatType(1) + FloatType(k) / 64});
    }
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    for (const Material *material : {&vacuum, &media, &crowded}) {
        for (const CudaKernel kernel : KERNELS) {
            for (const Block block : {Block{32, 8}, Block{7, 5}}) {
                INFO("material " << (material == &vacuum  ? "vacuum"
                                     : material == &media ? "media"
                                                          : "crowded")
                                 << ", kernel " << static_cast<int>(kernel) << ", block " << block.x
                                 << "x" << block.y);
                auto options = cuda_options(kernel, block);
                const Run arrays = run_material(BackendType::Cuda, options, *material, boundary);
                options.cuda_coeffs = fdtd::TMz::CudaCoeffs::Table;
                const Run table = run_material(BackendType::Cuda, options, *material, boundary);
                CHECK(table.fields == arrays.fields);
                CHECK(table.probes == arrays.probes);
            }
        }
    }
}

// Verify that the table matches serial within the bound, with graphs too.
TEST_CASE("CUDA coefficient table matches serial", "[backend][cuda]")
{
    require_cuda();
    const auto size = GENERATE(Size{64, 64}, Size{67, 45});
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Run serial = run(BackendType::Serial, {}, size, boundary);
    for (const CudaKernel kernel : KERNELS) {
        auto options = cuda_options(kernel, {32, 8});
        options.cuda_coeffs = fdtd::TMz::CudaCoeffs::Table;
        options.cuda_graph = true;
        const Run cuda = run(BackendType::Cuda, options, size, boundary);
        CHECK(relative_difference(serial.fields, cuda.fields) <= bound());
        CHECK(relative_difference(serial.probes, cuda.probes) <= bound());
    }
}

// Verify the traffic model of the table: on vacuum the core, the interior less its last column
// and row, shares a tuple and reads no coefficients, 6 words a cell fewer than the arrays, and
// no index.
TEST_CASE("CUDA models the traffic of the coefficient table", "[unit][backend][cuda]")
{
    require_cuda();
    for (const Boundary boundary : {Boundary::Pec, Boundary::Pmc, Boundary::Pml}) {
        INFO("boundary " << static_cast<int>(boundary));
        Simulation::Options options;
        options.backend = BackendType::Cuda;
        options.boundary = boundary;
        options.pml_thickness = 2;
        const Simulation arrays(8, 6, options);
        options.cuda_coeffs = fdtd::TMz::CudaCoeffs::Table;
        const Simulation table(8, 6, options);
        const double core = boundary == Boundary::Pml ? 3 : 15;
        CHECK(table.step_traffic().bytes ==
              arrays.step_traffic().bytes - 6 * core * static_cast<double>(sizeof(FloatType)));
        CHECK(table.step_traffic().flops == arrays.step_traffic().flops);
    }
}

// Verify that the tiled kernel gives the fused kernel's results bit for bit: H sources on the
// column and row a tile takes from its neighbours, every step count from 1 to 5 so the arrays
// end swapped and not, and probes on both H components.
TEST_CASE("CUDA tiled kernel matches the fused kernel bit for bit", "[backend][cuda]")
{
    require_cuda();
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Grid grid(70, 45, FloatType(1e-3), FloatType(2e-3));
    Material material(grid);
    material.set_pml(6, 3, 1e-6);
    material.fill_circle(40, 20, 6, {.eps_r = 3, .sigma_e = FloatType(0.2)});
    const Coeffs coeffs(material, fdtd::TMz::courant_dt(grid, 0.9));
    const std::vector<PointInjection> points{{Component::Hy, grid.index(31, 20)},
                                             {Component::Hx, grid.index(20, 7)},
                                             {Component::Hx, grid.index(31, 15)},
                                             {Component::Hy, grid.index(31, 15)},
                                             {Component::Ez, grid.index(50, 30)}};
    const std::vector<PointProbe> probes{{Component::Hy, grid.index(31, 20)},
                                         {Component::Hx, grid.index(20, 8)},
                                         {Component::Ez, grid.index(32, 16)}};
    for (const bool table : {false, true}) {
        for (std::size_t steps = 1; steps <= 5; ++steps) {
            INFO("table " << table << ", steps " << steps);
            Run runs[2];
            for (const CudaKernel kernel : {CudaKernel::Fused, CudaKernel::Tiled}) {
                auto options = cuda_options(kernel, {32, 8});
                if (table) {
                    options.cuda_coeffs = fdtd::TMz::CudaCoeffs::Table;
                }
                Fields fields(grid);
                fill_distinct(fields);
                const auto backend = fdtd::TMz::make_backend(BackendType::Cuda, grid, options);
                backend->upload(fields);
                backend->upload(coeffs);
                backend->set_boundary(boundary);
                backend->set_probes(probes);
                std::vector<FloatType> values(steps * points.size());
                for (std::size_t k = 0; k < values.size(); ++k) {
                    values[k] = FloatType(1 + k % 4) / FloatType(4);
                }
                backend->step(steps, std::span(points), std::span(values));
                backend->download(fields);
                Run &run = runs[kernel == CudaKernel::Tiled ? 1 : 0];
                append_fields(run.fields, fields);
                run.probes = backend->fetch_probes();
            }
            CHECK(runs[1].fields == runs[0].fields);
            CHECK(runs[1].probes == runs[0].probes);
        }
    }
}

// Verify the traffic model of the tiled kernel: one pass reads and writes each field once.
TEST_CASE("CUDA models the traffic of the tiled kernel", "[unit][backend][cuda]")
{
    require_cuda();
    Simulation::Options options;
    options.backend = BackendType::Cuda;
    options.boundary = Boundary::Pml;
    options.pml_thickness = 2;
    options.cuda_kernel = CudaKernel::Tiled;
    const Simulation tiled(8, 6, options);
    const double word = sizeof(FloatType);
    CHECK(tiled.step_traffic().bytes == (6 * 48 + 2 * 82 + 2 * 8 + 8 * 16 + 2 * 24) * word);
    CHECK(tiled.step_traffic().flops == 5 * 82 + 8 * 8 + 16 * 11);
    options.cuda_coeffs = fdtd::TMz::CudaCoeffs::Table;
    const Simulation table(8, 6, options);
    CHECK(table.step_traffic().bytes == tiled.step_traffic().bytes - 6 * 3 * word);
}

// Verify that snapshots copied out while the steps go on equal those of a run that stops for
// each, bit for bit: every kernel, with and without graphs, snapshots on consecutive steps (the
// two slots taking turns), across a graph's span and on the last step, over two runs.
TEST_CASE("CUDA async snapshots match the stopping ones bit for bit", "[backend][cuda]")
{
    require_cuda();
    struct Taken
    {
        std::vector<std::size_t> steps;
        std::vector<FloatType> fields;
        std::vector<FloatType> probes;
    };
    const auto boundary = GENERATE(Boundary::Pmc, Boundary::Pml);
    for (const CudaKernel kernel : KERNELS) {
        for (const bool graph : {false, true}) {
            INFO("kernel " << static_cast<int>(kernel) << ", graph " << graph);
            Taken taken[2];
            for (const bool async : {false, true}) {
                auto options = cuda_options(kernel, {32, 8});
                options.backend = BackendType::Cuda;
                options.boundary = boundary;
                options.pml_thickness = 6;
                options.cuda_graph = graph;
                options.cuda_async_snapshots = async;
                Simulation sim(53, 41, options);
                sim.add_source(20, 20, Component::Ez, fdtd::TMz::Gaussian::from_steps(sim.dt(), 6));
                sim.add_source(30, 12, Component::Hy, fdtd::TMz::Gaussian::from_steps(sim.dt(), 6));
                sim.add_probe(25, 25, Component::Ez);
                for (const std::size_t step :
                     std::vector<std::size_t>{1, 2, 3, 20, 37, 40, 41, 60}) {
                    sim.add_snapshot(step);
                }
                Taken &into = taken[async ? 1 : 0];
                const auto keep = [&](std::size_t step, const Fields &fields) {
                    into.steps.push_back(step);
                    append_fields(into.fields, fields);
                };
                sim.run(40, keep);
                sim.run(20, keep);
                sim.download();
                append_fields(into.fields, sim.fields());
                append(into.probes, sim.probe_data(0));
            }
            CHECK(taken[0].steps == std::vector<std::size_t>{1, 2, 3, 20, 37, 40, 41, 60});
            CHECK(taken[1].steps == taken[0].steps);
            CHECK(taken[1].fields == taken[0].fields);
            CHECK(taken[1].probes == taken[0].probes);
        }
    }
}

// Verify that the backend has its name and rejects settings it cannot run.
TEST_CASE("CUDA backend checks its options", "[unit][backend][cuda]")
{
    require_cuda();
    const Grid grid(16, 16);
    Simulation::Options options;
    const auto backend = fdtd::TMz::make_backend(BackendType::Cuda, grid, options);
    CHECK(backend->name() == "cuda");
    CHECK(backend->type() == BackendType::Cuda);
    CHECK_THROWS_AS(backend->step(1), std::logic_error);

    options.cuda_block_x = 0;
    CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::Cuda, grid, options),
                    std::invalid_argument);
    options.cuda_block_x = 32;
    options.cuda_block_y = 0;
    CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::Cuda, grid, options),
                    std::invalid_argument);
    options.cuda_block_x = 64;
    options.cuda_block_y = 32;
    CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::Cuda, grid, options),
                    std::invalid_argument);
    options.cuda_block_x = 2048;
    options.cuda_block_y = 1;
    CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::Cuda, grid, options),
                    std::invalid_argument);
    options.cuda_block_x = 32;
    options.cuda_block_y = 1;
    CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::Cuda, Grid(16, 70000), options),
                    std::invalid_argument);
    options.cuda_block_y = 8;
    CHECK_THROWS_AS(fdtd::TMz::make_backend(BackendType::Cuda, Grid(50000, 50000), options),
                    std::invalid_argument);
}
