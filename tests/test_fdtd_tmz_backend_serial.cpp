/*
 * C++ file for the tests of the 2D TMz serial backend: its kernels, partitions and stepper.
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
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers under test.
#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_material.h"

// Standard library for math, bounds, sizes, errors and buffers.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Shorthand for the types under test.
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Coeffs;
using fdtd::TMz::Component;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Fields;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;
using fdtd::TMz::PointInjection;
using fdtd::TMz::PointProbe;
using fdtd::TMz::backends::serial::Block;
using fdtd::TMz::backends::serial::CoeffsView;
using fdtd::TMz::backends::serial::interior_block;
using fdtd::TMz::backends::serial::intersect;
using fdtd::TMz::backends::serial::Partition;
using fdtd::TMz::backends::serial::Stepper;
using fdtd::TMz::backends::serial::update_boundary_block;
using fdtd::TMz::backends::serial::update_e_block;
using fdtd::TMz::backends::serial::update_e_interior_block;
using fdtd::TMz::backends::serial::update_h_block;
using fdtd::TMz::backends::serial::update_pml_block;
using fdtd::TMz::backends::serial::whole_block;

// Relative tolerance that holds in single precision.
static constexpr double TOL = 1e-5;

// Fill every field with distinct non-zero values.
static void fill_distinct(Fields &fields)
{
    for (std::size_t k = 0; k < fields.grid().size(); ++k) {
        fields.ez()[k] = FloatType(1 + k);
        fields.hx()[k] = FloatType(0.5 + 0.25 * double(k));
        fields.hy()[k] = FloatType(-1 + 0.5 * double(k));
    }
}

// Update H over the whole grid.
static void update_h(Fields &fields, CoeffsView coeffs)
{
    update_h_block(fields, coeffs, whole_block(fields.grid()));
}

// Update the interior Ez over the whole grid, inside the PML when there is one.
static void update_e(Fields &fields, CoeffsView coeffs, std::size_t pml)
{
    update_e_interior_block(fields, coeffs, whole_block(fields.grid()), pml);
}

// Update Ezx, Ezy and Ez over the PML slabs of the whole grid.
static void update_pml(Fields &fields, CoeffsView coeffs, std::size_t pml)
{
    update_pml_block(fields, coeffs, whole_block(fields.grid()), pml);
}

// Update the edge Ez of the whole grid.
static void update_boundary(Fields &fields, CoeffsView coeffs, Boundary boundary)
{
    update_boundary_block(fields, coeffs, whole_block(fields.grid()), boundary);
}

// Run whole steps: H, interior E, then the boundary.
static void run(Fields &fields, const Coeffs &coeffs, Boundary boundary, int nsteps)
{
    for (int n = 0; n < nsteps; ++n) {
        update_h(fields, coeffs);
        update_e(fields, coeffs, 0);
        update_boundary(fields, coeffs, boundary);
    }
}

// Verify one step from a unit Ez impulse at S = 1/2: the centre to 1/2, each neighbour to 1/8.
TEST_CASE("Serial update of an Ez impulse", "[unit][serial_update]")
{
    const Grid grid(7, 7, 1e-3F, 1e-3F);
    const Material material(grid);
    const Coeffs coeffs(material, courant_dt(grid, 0.5));
    Fields fields(grid);
    const std::size_t c = grid.index(3, 3);
    fields.ez()[c] = 1;

    update_h(fields, coeffs);
    const double h = double(coeffs.coeff_hy_2()[c]) / double(grid.dx());
    CHECK_THAT(double(fields.hy()[grid.index(2, 3)]), WithinRel(h, TOL));
    CHECK_THAT(double(fields.hy()[grid.index(3, 3)]), WithinRel(-h, TOL));
    CHECK_THAT(double(fields.hx()[grid.index(3, 2)]), WithinRel(-h, TOL));
    CHECK_THAT(double(fields.hx()[grid.index(3, 3)]), WithinRel(h, TOL));
    CHECK(fields.hx()[grid.index(2, 3)] == FloatType(0));
    CHECK(fields.hy()[grid.index(3, 2)] == FloatType(0));

    update_e(fields, coeffs, 0);
    CHECK_THAT(double(fields.ez()[c]), WithinRel(0.5, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(2, 3)]), WithinRel(0.125, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(4, 3)]), WithinRel(0.125, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(3, 2)]), WithinRel(0.125, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(3, 4)]), WithinRel(0.125, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(2, 2)]), WithinAbs(0.0, 1e-12));
}

// Verify that each update reads its own coefficient arrays, with every curl factor zero.
TEST_CASE("Serial update reads the coefficients by field", "[unit][serial_update]")
{
    const Grid grid(4, 4);
    const std::vector<FloatType> zero(grid.size(), FloatType(0));
    const std::vector<FloatType> ezX(grid.size(), FloatType(0.5));
    const std::vector<FloatType> ezY(grid.size(), FloatType(8));
    const std::vector<FloatType> hx(grid.size(), FloatType(0.25));
    const std::vector<FloatType> hy(grid.size(), FloatType(0.125));
    const CoeffsView coeffs(ezX.data(), zero.data(), ezY.data(), zero.data(), hx.data(), zero.data(), hy.data(), zero.data());
    Fields fields(grid);
    for (std::size_t k = 0; k < grid.size(); ++k) {
        fields.ez()[k] = 1;
        fields.hx()[k] = 1;
        fields.hy()[k] = 1;
    }

    update_h(fields, coeffs);
    update_e(fields, coeffs, 0);

    CHECK(fields.hx()[grid.index(1, 1)] == FloatType(0.25));
    CHECK(fields.hy()[grid.index(1, 1)] == FloatType(0.125));
    CHECK(fields.ez()[grid.index(1, 1)] == FloatType(0.5));
    CHECK(fields.ez()[grid.index(2, 2)] == FloatType(0.5));
}

// Verify that the update leaves the edge Ez, last-row Hx and last-column Hy untouched.
TEST_CASE("Serial update leaves the edges alone", "[unit][serial_update]")
{
    const Grid grid(4, 3);
    const Material material(grid);
    const Coeffs coeffs(material, courant_dt(grid, 0.9));
    Fields fields(grid);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            fields.ez()[grid.index(i, j)] = FloatType(1 + i + 10 * j);
        }
    }
    const std::size_t lastRow = grid.ny() - 1;
    const std::size_t lastCol = grid.nx() - 1;

    update_h(fields, coeffs);
    update_e(fields, coeffs, 0);

    for (std::size_t i = 0; i < grid.nx(); ++i) {
        CHECK(fields.hx()[grid.index(i, lastRow)] == FloatType(0));
        CHECK(fields.ez()[grid.index(i, 0)] == FloatType(1 + i));
        CHECK(fields.ez()[grid.index(i, lastRow)] == FloatType(1 + i + 10 * lastRow));
    }
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        CHECK(fields.hy()[grid.index(lastCol, j)] == FloatType(0));
        CHECK(fields.ez()[grid.index(0, j)] == FloatType(1 + 10 * j));
        CHECK(fields.ez()[grid.index(lastCol, j)] == FloatType(1 + lastCol + 10 * j));
    }
}

// Verify that one-cell-wide grids update without touching memory outside them.
TEST_CASE("Serial update on degenerate grids", "[unit][serial_update]")
{
    for (const Grid &grid : {Grid(1, 1), Grid(1, 5), Grid(5, 1), Grid(2, 2)}) {
        const Material material(grid);
        const Coeffs coeffs(material, courant_dt(grid, 0.9));
        Fields fields(grid);
        fields.ez()[0] = 1;

        update_h(fields, coeffs);
        update_e(fields, coeffs, 0);

        CHECK(fields.ez()[0] == FloatType(1));
    }
}

// Verify that PEC zeroes every edge cell and leaves the interior alone.
TEST_CASE("PEC boundary zeroes the edge Ez", "[unit][boundary]")
{
    const Grid grid(5, 4);
    const Coeffs coeffs(Material(grid), courant_dt(grid, 0.9));
    Fields fields(grid);
    fill_distinct(fields);

    update_boundary(fields, coeffs, Boundary::Pec);

    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            const std::size_t k = grid.index(i, j);
            const bool edge = i == 0 || j == 0 || i + 1 == grid.nx() || j + 1 == grid.ny();
            CHECK(fields.ez()[k] == (edge ? FloatType(0) : FloatType(1 + k)));
        }
    }
}

// Verify PMC at a corner and the right edge against hand values, H outside read as zero.
TEST_CASE("PMC boundary reads H outside as zero", "[unit][boundary]")
{
    const Grid grid(3, 3, 1e-3F, 2e-3F);
    const Coeffs coeffs(Material(grid), courant_dt(grid, 0.9));
    Fields fields(grid);
    fill_distinct(fields);
    const auto ez = [&](std::size_t i, std::size_t j) {
        return double(fields.ez()[grid.index(i, j)]);
    };
    const auto hx = [&](std::size_t i, std::size_t j) {
        return double(fields.hx()[grid.index(i, j)]);
    };
    const auto hy = [&](std::size_t i, std::size_t j) {
        return double(fields.hy()[grid.index(i, j)]);
    };
    const double ce1 = double(coeffs.coeff_ez_x1()[0]);
    const double ce2 = double(coeffs.coeff_ez_x2()[0]);
    const double dx = double(grid.dx());
    const double dy = double(grid.dy());

    const double corner = ce1 * ez(0, 0) + ce2 * (hy(0, 0) / dx - hx(0, 0) / dy);
    const double right = ce1 * ez(2, 1) + ce2 * ((0 - hy(1, 1)) / dx - (hx(2, 1) - hx(2, 0)) / dy);
    const double top = ce1 * ez(1, 2) + ce2 * ((hy(1, 2) - hy(0, 2)) / dx - (0 - hx(1, 1)) / dy);
    const double centre = ez(1, 1);

    update_boundary(fields, coeffs, Boundary::Pmc);

    CHECK_THAT(ez(0, 0), WithinRel(corner, TOL));
    CHECK_THAT(ez(2, 1), WithinRel(right, TOL));
    CHECK_THAT(ez(1, 2), WithinRel(top, TOL));
    CHECK(ez(1, 1) == centre);
}

// Verify that an impulse stays symmetric under PMC on square cells and reaches the edge.
TEST_CASE("PMC boundary keeps a centred impulse symmetric", "[unit][boundary]")
{
    const Grid grid(9, 9, 1e-3F, 1e-3F);
    const Coeffs coeffs(Material(grid), courant_dt(grid, 0.7));
    Fields pmc(grid);
    Fields pec(grid);
    pmc.ez()[grid.index(4, 4)] = 1;
    pec.ez()[grid.index(4, 4)] = 1;

    run(pmc, coeffs, Boundary::Pmc, 20);
    run(pec, coeffs, Boundary::Pec, 20);

    const std::size_t n = grid.nx() - 1;
    for (std::size_t j = 0; j <= n; ++j) {
        for (std::size_t i = 0; i <= n; ++i) {
            const double v = double(pmc.ez()[grid.index(i, j)]);
            CHECK_THAT(double(pmc.ez()[grid.index(n - i, j)]), WithinAbs(v, 1e-6));
            CHECK_THAT(double(pmc.ez()[grid.index(i, n - j)]), WithinAbs(v, 1e-6));
            CHECK_THAT(double(pmc.ez()[grid.index(j, i)]), WithinAbs(v, 1e-6));
        }
        CHECK(pec.ez()[grid.index(0, j)] == FloatType(0));
    }
    CHECK(pmc.ez()[grid.index(0, 4)] != FloatType(0));
}

// Verify that both conditions handle grids one or two cells wide.
TEST_CASE("Boundary on degenerate grids", "[unit][boundary]")
{
    for (const Grid &grid : {Grid(1, 1), Grid(1, 4), Grid(4, 1), Grid(2, 2)}) {
        const Coeffs coeffs(Material(grid), courant_dt(grid, 0.9));
        Fields fields(grid);
        fill_distinct(fields);

        update_boundary(fields, coeffs, Boundary::Pmc);
        update_boundary(fields, coeffs, Boundary::Pec);

        for (const FloatType v : fields.ez()) {
            CHECK(v == FloatType(0));
        }
    }
}

// Verify that the interior and split updates together cover every non-edge cell once.
TEST_CASE("Split update matches the plain update without loss", "[unit][serial_update][pml]")
{
    const std::size_t pml = 3;
    const Grid grid(12, 10);
    const Coeffs coeffs(Material(grid), courant_dt(grid, 0.9));
    Fields plain(grid);
    Fields split(grid);
    fill_distinct(plain);
    fill_distinct(split);
    for (std::size_t k = 0; k < grid.size(); ++k) {
        split.ezx()[k] = split.ez()[k];
    }

    update_e(plain, coeffs, 0);
    update_e(split, coeffs, pml);
    update_pml(split, coeffs, pml);

    double scale = 0;
    for (std::size_t k = 0; k < grid.size(); ++k) {
        scale = std::max(scale, std::abs(double(plain.ez()[k])));
    }
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            const std::size_t k = grid.index(i, j);
            const bool edge = i == 0 || j == 0 || i + 1 == grid.nx() || j + 1 == grid.ny();
            const bool inside = i >= pml && j >= pml && i + pml < grid.nx() && j + pml < grid.ny();
            CHECK_THAT(double(split.ez()[k]), WithinAbs(double(plain.ez()[k]), TOL * scale));
            if (edge || inside) {
                CHECK(split.ezx()[k] == FloatType(1 + k));
                CHECK(split.ezy()[k] == FloatType(0));
            } else {
                CHECK(split.ezx()[k] != FloatType(1 + k));
                CHECK(split.ez()[k] == split.ezx()[k] + split.ezy()[k]);
            }
        }
    }
}

// Verify that each split part decays with the conductivity along its own direction.
TEST_CASE("Split update damps each part along its direction", "[unit][serial_update][pml]")
{
    const Grid grid(20, 16);
    Material material(grid);
    material.set_pml(4, 3, 1e-6);
    const Coeffs coeffs(material, courant_dt(grid, 0.9));
    Fields fields(grid);
    for (std::size_t k = 0; k < grid.size(); ++k) {
        fields.ezx()[k] = 1;
        fields.ezy()[k] = 1;
    }

    update_pml(fields, coeffs, 4);

    const Fields &f = fields;
    for (const std::size_t i : {std::size_t(1), std::size_t(18)}) {
        CHECK(f.ezx()[grid.index(i, 8)] < FloatType(1));
        CHECK(f.ezy()[grid.index(i, 8)] == FloatType(1));
    }
    for (const std::size_t j : {std::size_t(1), std::size_t(14)}) {
        CHECK(f.ezx()[grid.index(10, j)] == FloatType(1));
        CHECK(f.ezy()[grid.index(10, j)] < FloatType(1));
    }
    CHECK(f.ezx()[grid.index(1, 1)] < FloatType(1));
    CHECK(f.ezy()[grid.index(1, 1)] < FloatType(1));
    CHECK(f.ezx()[grid.index(10, 8)] == FloatType(1));
    CHECK(f.ezy()[grid.index(10, 8)] == FloatType(1));
}

// Verify that PEC zeroes the split parts on the edge along with Ez.
TEST_CASE("PEC boundary zeroes the edge split fields", "[unit][boundary][pml]")
{
    const Grid grid(6, 5);
    const Coeffs coeffs(Material(grid), courant_dt(grid, 0.9));
    Fields fields(grid);
    for (std::size_t k = 0; k < grid.size(); ++k) {
        fields.ezx()[k] = 1;
        fields.ezy()[k] = 2;
    }

    update_boundary(fields, coeffs, Boundary::Pml);

    CHECK(fields.ezx()[grid.index(0, 2)] == FloatType(0));
    CHECK(fields.ezy()[grid.index(5, 4)] == FloatType(0));
    CHECK(fields.ezx()[grid.index(3, 0)] == FloatType(0));
    CHECK(fields.ezy()[grid.index(2, 2)] == FloatType(2));
}

// Helpers private to this file.
namespace
{
    // Fill every field with distinct values that keep a few steps bounded.
    void fill_bounded(Fields &fields)
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

    // Cut a grid into blocks of at most w by h cells.
    std::vector<Block> cut(const Grid &grid, std::size_t w, std::size_t h)
    {
        std::vector<Block> blocks;
        for (std::size_t j = 0; j < grid.ny(); j += h) {
            for (std::size_t i = 0; i < grid.nx(); i += w) {
                blocks.push_back({
                    .i0 = i,
                    .i1 = std::min(i + w, grid.nx()),
                    .j0 = j,
                    .j1 = std::min(j + h, grid.ny()),
                });
            }
        }
        return blocks;
    }

    // Check that two sets of fields hold the same bits.
    bool same_fields(const Fields &a, const Fields &b)
    {
        return std::equal(a.ez().begin(), a.ez().end(), b.ez().begin()) && std::equal(a.hx().begin(), a.hx().end(), b.hx().begin()) &&
               std::equal(a.hy().begin(), a.hy().end(), b.hy().begin()) && std::equal(a.ezx().begin(), a.ezx().end(), b.ezx().begin()) &&
               std::equal(a.ezy().begin(), a.ezy().end(), b.ezy().begin());
    }

    // Run steps over the whole grid, or block by block.
    void run_blocks(Fields &fields, const Coeffs &coeffs, const std::vector<Block> &blocks, std::size_t pml, Boundary boundary, int nsteps)
    {
        for (int n = 0; n < nsteps; ++n) {
            for (const Block &block : blocks) {
                update_h_block(fields, coeffs, block);
            }
            for (const Block &block : blocks) {
                update_e_block(fields, coeffs, block, pml, boundary);
            }
        }
    }
}

// Verify the whole block, intersection and emptiness.
TEST_CASE("Blocks intersect", "[unit][block]")
{
    const Grid grid(7, 5);
    const Block whole = whole_block(grid);
    CHECK(whole.i0 == 0);
    CHECK(whole.i1 == 7);
    CHECK(whole.j0 == 0);
    CHECK(whole.j1 == 5);
    CHECK_FALSE(whole.empty());

    const Block part = intersect(whole, {.i0 = 5, .i1 = 9, .j0 = 1, .j1 = 3});
    CHECK(part.i0 == 5);
    CHECK(part.i1 == 7);
    CHECK(part.j0 == 1);
    CHECK(part.j1 == 3);
    CHECK(intersect(whole, {.i0 = 7, .i1 = 9, .j0 = 0, .j1 = 5}).empty());
    CHECK(Block{.i0 = 2, .i1 = 2, .j0 = 0, .j1 = 1}.empty());
}

// Verify the interior block with and without a PML.
TEST_CASE("Interior block leaves out the edge and the PML", "[unit][block]")
{
    const Grid grid(12, 9);
    const Block plain = interior_block(grid, 0);
    CHECK(plain.i0 == 1);
    CHECK(plain.i1 == 11);
    CHECK(plain.j0 == 1);
    CHECK(plain.j1 == 8);
    const Block pml = interior_block(grid, 3);
    CHECK(pml.i0 == 3);
    CHECK(pml.i1 == 9);
    CHECK(pml.j0 == 3);
    CHECK(pml.j1 == 6);
}

// Verify that any cut into blocks updates the fields as the whole grid does, bit for bit.
TEST_CASE("Block updates match the whole-grid update", "[unit][block]")
{
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const auto w = GENERATE(std::size_t(1), std::size_t(5), std::size_t(64));
    const auto h = GENERATE(std::size_t(1), std::size_t(3), std::size_t(64));
    const Grid grid(23, 19);
    const std::size_t pml = boundary == Boundary::Pml ? 4 : 0;
    Material material(grid);
    material.set_pml(pml, 3, 1e-6);
    const Coeffs coeffs(material, fdtd::TMz::courant_dt(grid, 0.9));
    Fields whole(grid);
    fill_bounded(whole);
    Fields blocked(grid);
    fill_bounded(blocked);

    run_blocks(whole, coeffs, {whole_block(grid)}, pml, boundary, 4);
    run_blocks(blocked, coeffs, cut(grid, w, h), pml, boundary, 4);

    CHECK(same_fields(whole, blocked));
}

// Verify that the edge update changes only the edge cells of its block.
TEST_CASE("Edge update stays inside its block", "[unit][block][boundary]")
{
    const Grid grid(6, 5);
    const Coeffs coeffs(Material(grid), fdtd::TMz::courant_dt(grid, 0.9));
    Fields fields(grid);
    std::fill(fields.ez().begin(), fields.ez().end(), FloatType(1));

    update_boundary_block(fields, coeffs, {.i0 = 0, .i1 = 3, .j0 = 0, .j1 = 2}, Boundary::Pec);

    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            const bool zeroed = (j == 0 && i < 3) || (i == 0 && j < 2);
            CHECK(fields.ez()[grid.index(i, j)] == FloatType(zeroed ? 0 : 1));
        }
    }
}

// Verify that a block on a one-cell-wide grid visits each edge cell once.
TEST_CASE("Block updates on degenerate grids", "[unit][block]")
{
    const auto nx = GENERATE(std::size_t(1), std::size_t(2), std::size_t(5));
    const auto ny = GENERATE(std::size_t(1), std::size_t(2), std::size_t(4));
    const Grid grid(nx, ny);
    const Coeffs coeffs(Material(grid), fdtd::TMz::courant_dt(grid, 0.9));
    Fields whole(grid);
    fill_bounded(whole);
    Fields blocked(grid);
    fill_bounded(blocked);

    run_blocks(whole, coeffs, {whole_block(grid)}, 0, Boundary::Pmc, 3);
    run_blocks(blocked, coeffs, cut(grid, 1, 1), 0, Boundary::Pmc, 3);

    CHECK(same_fields(whole, blocked));
}

// Helpers private to this file.
namespace
{
    // Check that every cell lies in exactly one block of one part, the part owner() names.
    void check_cover(const Partition &partition, const Grid &grid)
    {
        std::vector<int> count(grid.nx() * grid.ny(), 0);
        std::vector<std::size_t> part(grid.nx() * grid.ny(), 0);
        for (std::size_t p = 0; p < partition.size(); ++p) {
            for (const Block &block : partition.blocks(p)) {
                CHECK_FALSE(block.empty());
                for (std::size_t j = block.j0; j < block.j1; ++j) {
                    for (std::size_t i = block.i0; i < block.i1; ++i) {
                        ++count[j * grid.nx() + i];
                        part[j * grid.nx() + i] = p;
                    }
                }
            }
        }
        for (std::size_t j = 0; j < grid.ny(); ++j) {
            for (std::size_t i = 0; i < grid.nx(); ++i) {
                CHECK(count[j * grid.nx() + i] == 1);
                CHECK(partition.owner(i, j) == part[j * grid.nx() + i]);
            }
        }
    }
}

// Verify that the whole partition is one part, one block.
TEST_CASE("Whole partition is the grid", "[unit][partition]")
{
    const Grid grid(7, 5);
    const Partition partition = Partition::whole(grid);
    REQUIRE(partition.size() == 1);
    REQUIRE(partition.blocks(0).size() == 1);
    CHECK(partition.blocks(0)[0].i1 == 7);
    CHECK(partition.blocks(0)[0].j1 == 5);
    CHECK(partition.tiles_x() == 1);
    CHECK(partition.tiles_y() == 1);
    check_cover(partition, grid);
}

// Verify that tiles cover the grid row-major, the last ones clipped.
TEST_CASE("Tiles cover the grid", "[unit][partition]")
{
    const auto w = GENERATE(std::size_t(1), std::size_t(4), std::size_t(7), std::size_t(50));
    const auto h = GENERATE(std::size_t(1), std::size_t(3), std::size_t(50));
    const Grid grid(23, 10);
    const Partition partition = Partition::tiles(grid, w, h);
    CHECK(partition.tiles_x() == (23 + w - 1) / w);
    CHECK(partition.tiles_y() == (10 + h - 1) / h);
    CHECK(partition.size() == partition.tiles_x() * partition.tiles_y());
    CHECK(partition.blocks(1 % partition.size())[0].i0 == (partition.tiles_x() > 1 ? w : 0));
    check_cover(partition, grid);
}

// Verify that empty tiles are rejected.
TEST_CASE("Partitions reject empty pieces", "[unit][partition]")
{
    const Grid grid(4, 4);
    CHECK_THROWS_AS(Partition::tiles(grid, 0, 2), std::invalid_argument);
    CHECK_THROWS_AS(Partition::tiles(grid, 2, 0), std::invalid_argument);
}

// Helpers private to this file.
namespace
{
    // Steps of every run.
    constexpr std::size_t STEPS = 12;

    // Result of a run: the fields and the probe records.
    struct Run
    {
        /** Fields after the run. */
        Fields fields;

        /** Probe records, step-major. */
        std::vector<FloatType> records;
    };

    // Run the stepper part by part over a partition: every H phase, then every E phase.
    Run run_parts(const Grid &grid, const Partition &partition, Boundary boundary)
    {
        const std::size_t pml = boundary == Boundary::Pml ? 4 : 0;
        Material material(grid);
        material.set_pml(pml, 3, 1e-6);
        const Coeffs coeffs(material, fdtd::TMz::courant_dt(grid, 0.9));
        const std::vector<PointInjection> sources{
            {Component::Ez, grid.index(9, 8)}, {Component::Hx, grid.index(12, 6)}, {Component::Hy, grid.index(6, 11)}};
        std::vector<FloatType> values(STEPS * sources.size());
        for (std::size_t k = 0; k < values.size(); ++k) {
            values[k] = FloatType(1 + k % 5);
        }
        const std::vector<PointProbe> probes{{Component::Ez, grid.index(10, 9)},
                                             {Component::Hy, grid.index(0, 0)},
                                             {Component::Ez, grid.index(10, 9)},
                                             {Component::Hx, grid.index(18, 13)}};
        Run result{Fields(grid), std::vector<FloatType>(STEPS * probes.size())};
        const Stepper stepper(result.fields, coeffs, pml, boundary, partition, sources, values, probes, result.records.data());
        for (std::size_t n = 0; n < STEPS; ++n) {
            for (std::size_t p = 0; p < partition.size(); ++p) {
                stepper.h_phase(p, n);
            }
            for (std::size_t p = 0; p < partition.size(); ++p) {
                stepper.e_phase(p, n);
            }
        }
        return result;
    }

    // Check that two runs hold the same bits.
    bool same_run(const Run &a, const Run &b)
    {
        return std::equal(a.fields.ez().begin(), a.fields.ez().end(), b.fields.ez().begin()) &&
               std::equal(a.fields.hx().begin(), a.fields.hx().end(), b.fields.hx().begin()) &&
               std::equal(a.fields.hy().begin(), a.fields.hy().end(), b.fields.hy().begin()) &&
               std::equal(a.fields.ezx().begin(), a.fields.ezx().end(), b.fields.ezx().begin()) &&
               std::equal(a.fields.ezy().begin(), a.fields.ezy().end(), b.fields.ezy().begin()) && a.records == b.records;
    }
}

// Verify that sources and probes owned by parts give the whole-grid run, bit for bit.
TEST_CASE("Stepper over parts matches the whole grid", "[unit][stepper]")
{
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const Grid grid(21, 17);
    const Run whole = run_parts(grid, Partition::whole(grid), boundary);

    CHECK(same_run(whole, run_parts(grid, Partition::tiles(grid, grid.nx(), 1), boundary)));
    CHECK(same_run(whole, run_parts(grid, Partition::tiles(grid, 5, 4), boundary)));
    CHECK(same_run(whole, run_parts(grid, Partition::tiles(grid, 1, 1), boundary)));
}

// Verify that the records are step-major, a repeated probe twice, and the sources land.
TEST_CASE("Stepper writes its records step-major", "[unit][stepper]")
{
    const Grid grid(21, 17);
    const Run result = run_parts(grid, Partition::tiles(grid, 4, 4), Boundary::Pec);

    REQUIRE(result.records.size() == STEPS * 4);
    for (std::size_t n = 0; n < STEPS; ++n) {
        CHECK(result.records[n * 4] == result.records[n * 4 + 2]);
        CHECK(result.records[n * 4 + 1] == FloatType(0));
    }
    CHECK(result.records[(STEPS - 1) * 4] == result.fields.ez()[grid.index(10, 9)]);
    CHECK(result.records[(STEPS - 1) * 4 + 3] == result.fields.hx()[grid.index(18, 13)]);
    CHECK(result.fields.ez()[grid.index(9, 8)] != FloatType(0));
}
