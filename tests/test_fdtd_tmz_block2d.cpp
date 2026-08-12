/*
 * C++ file for the tests of the 2D TMz block kernels.
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
#include "fdtd/TMz/backends/fdtd_tmz_block2d.h"
#include "fdtd/TMz/fdtd_tmz_grid2d.h"
#include "fdtd/TMz/fdtd_tmz_material2d.h"

// Standard library for algorithms, sizes and buffers.
#include <algorithm>
#include <cstddef>
#include <vector>

// Shorthand for the types under test.
using fdtd::FloatType;
using fdtd::TMz::Block2D;
using fdtd::TMz::Boundary;
using fdtd::TMz::Coeffs2D;
using fdtd::TMz::Fields2D;
using fdtd::TMz::Grid2D;
using fdtd::TMz::Material2D;

// Helpers private to this file.
namespace
{
    // Fill every field with distinct values that keep a few steps bounded.
    void fill_distinct(Fields2D &fields)
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
    std::vector<Block2D> cut(const Grid2D &grid, std::size_t w, std::size_t h)
    {
        std::vector<Block2D> blocks;
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
    bool same(const Fields2D &a, const Fields2D &b)
    {
        return std::equal(a.ez().begin(), a.ez().end(), b.ez().begin()) &&
               std::equal(a.hx().begin(), a.hx().end(), b.hx().begin()) &&
               std::equal(a.hy().begin(), a.hy().end(), b.hy().begin()) &&
               std::equal(a.ezx().begin(), a.ezx().end(), b.ezx().begin()) &&
               std::equal(a.ezy().begin(), a.ezy().end(), b.ezy().begin());
    }

    // Run steps over the whole grid, or block by block, with or without SIMD.
    template <bool Simd>
    void run(Fields2D &fields, const Coeffs2D &coeffs, const std::vector<Block2D> &blocks,
             std::size_t pml, Boundary boundary, int nsteps)
    {
        for (int n = 0; n < nsteps; ++n) {
            for (const Block2D &block : blocks) {
                fdtd::TMz::update_h_block<Simd>(fields, coeffs, block);
            }
            for (const Block2D &block : blocks) {
                fdtd::TMz::update_e_block<Simd>(fields, coeffs, block, pml, boundary);
            }
        }
    }
}

// Verify the whole block, intersection and emptiness.
TEST_CASE("Blocks intersect", "[unit][block2d]")
{
    const Grid2D grid(7, 5);
    const Block2D whole = fdtd::TMz::whole_block(grid);
    CHECK(whole.i0 == 0);
    CHECK(whole.i1 == 7);
    CHECK(whole.j0 == 0);
    CHECK(whole.j1 == 5);
    CHECK_FALSE(whole.empty());

    const Block2D part = fdtd::TMz::intersect(whole, {.i0 = 5, .i1 = 9, .j0 = 1, .j1 = 3});
    CHECK(part.i0 == 5);
    CHECK(part.i1 == 7);
    CHECK(part.j0 == 1);
    CHECK(part.j1 == 3);
    CHECK(fdtd::TMz::intersect(whole, {.i0 = 7, .i1 = 9, .j0 = 0, .j1 = 5}).empty());
    CHECK(Block2D{.i0 = 2, .i1 = 2, .j0 = 0, .j1 = 1}.empty());
}

// Verify the interior block with and without a PML.
TEST_CASE("Interior block leaves out the edge and the PML", "[unit][block2d]")
{
    const Grid2D grid(12, 9);
    const Block2D plain = fdtd::TMz::interior_block(grid, 0);
    CHECK(plain.i0 == 1);
    CHECK(plain.i1 == 11);
    CHECK(plain.j0 == 1);
    CHECK(plain.j1 == 8);
    const Block2D pml = fdtd::TMz::interior_block(grid, 3);
    CHECK(pml.i0 == 3);
    CHECK(pml.i1 == 9);
    CHECK(pml.j0 == 3);
    CHECK(pml.j1 == 6);
}

// Verify that any cut into blocks updates the fields as the whole grid does, bit for bit.
TEST_CASE("Block updates match the whole-grid update", "[unit][block2d]")
{
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const auto w = GENERATE(std::size_t(1), std::size_t(5), std::size_t(64));
    const auto h = GENERATE(std::size_t(1), std::size_t(3), std::size_t(64));
    const Grid2D grid(23, 19);
    const std::size_t pml = boundary == Boundary::Pml ? 4 : 0;
    Material2D material(grid);
    material.set_pml(pml, 3, 1e-6);
    const Coeffs2D coeffs(material, fdtd::TMz::courant_dt(grid, 0.9));
    Fields2D whole(grid);
    fill_distinct(whole);
    Fields2D blocked(grid);
    fill_distinct(blocked);
    Fields2D simd(grid);
    fill_distinct(simd);

    run<false>(whole, coeffs, {fdtd::TMz::whole_block(grid)}, pml, boundary, 4);
    run<false>(blocked, coeffs, cut(grid, w, h), pml, boundary, 4);
    run<true>(simd, coeffs, cut(grid, w, h), pml, boundary, 4);

    CHECK(same(whole, blocked));
    CHECK(same(whole, simd));
}

// Verify that the edge update changes only the edge cells of its block.
TEST_CASE("Edge update stays inside its block", "[unit][block2d][boundary]")
{
    const Grid2D grid(6, 5);
    const Coeffs2D coeffs(Material2D(grid), fdtd::TMz::courant_dt(grid, 0.9));
    Fields2D fields(grid);
    std::fill(fields.ez().begin(), fields.ez().end(), FloatType(1));

    fdtd::TMz::update_boundary_block(fields, coeffs, {.i0 = 0, .i1 = 3, .j0 = 0, .j1 = 2},
                                     Boundary::Pec);

    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            const bool zeroed = (j == 0 && i < 3) || (i == 0 && j < 2);
            CHECK(fields.ez()[grid.index(i, j)] == FloatType(zeroed ? 0 : 1));
        }
    }
}

// Verify that a block on a one-cell-wide grid visits each edge cell once.
TEST_CASE("Block updates on degenerate grids", "[unit][block2d]")
{
    const auto nx = GENERATE(std::size_t(1), std::size_t(2), std::size_t(5));
    const auto ny = GENERATE(std::size_t(1), std::size_t(2), std::size_t(4));
    const Grid2D grid(nx, ny);
    const Coeffs2D coeffs(Material2D(grid), fdtd::TMz::courant_dt(grid, 0.9));
    Fields2D whole(grid);
    fill_distinct(whole);
    Fields2D blocked(grid);
    fill_distinct(blocked);

    run<false>(whole, coeffs, {fdtd::TMz::whole_block(grid)}, 0, Boundary::Pmc, 3);
    run<false>(blocked, coeffs, cut(grid, 1, 1), 0, Boundary::Pmc, 3);

    CHECK(same(whole, blocked));
}
