/*
 * C++ file for the tests of the 2D TMz update of one cell, run on the host.
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

// Project headers under test, and the serial kernels they are checked against.
#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_cell.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_material.h"

// Standard library for algorithms, math, limits, sizes and views.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

// Shorthand for the types under test.
using fdtd::FloatType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Coeffs;
using fdtd::TMz::Fields;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;
using fdtd::TMz::backends::gpu::DeviceCoeffs;
using fdtd::TMz::backends::gpu::DeviceFields;

// Helpers private to this file.
namespace
{
    // Steps of every run.
    constexpr int STEPS = 6;

    // Bound of the difference from the CPU kernels, in machine epsilons per step, from the build.
    constexpr double TOLERANCE = FDTD_TEST_TOLERANCE;

    // Grid of a run with its PML thickness.
    struct Case
    {
        /** Cells along x. */
        std::size_t nx;

        /** Cells along y. */
        std::size_t ny;

        /** PML thickness under Boundary::Pml. */
        std::size_t pml;
    };

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

    // View host fields as the cell functions take them.
    DeviceFields view(Fields &fields, std::size_t pml)
    {
        const Grid &grid = fields.grid();
        return {
            .ez = fields.ez().data(),
            .hx = fields.hx().data(),
            .hy = fields.hy().data(),
            .ezx = fields.ezx().data(),
            .ezy = fields.ezy().data(),
            .nx = static_cast<int>(grid.nx()),
            .ny = static_cast<int>(grid.ny()),
            .pitch = static_cast<int>(grid.pitch()),
            .inv_dx = FloatType(1) / grid.dx(),
            .inv_dy = FloatType(1) / grid.dy(),
            .pml = static_cast<int>(pml),
        };
    }

    // View host coefficients as the cell functions take them.
    DeviceCoeffs view(const Coeffs &coeffs)
    {
        return {
            .coeff_ez_x1 = coeffs.coeff_ez_x1().data(),
            .coeff_ez_x2 = coeffs.coeff_ez_x2().data(),
            .coeff_ez_y1 = coeffs.coeff_ez_y1().data(),
            .coeff_ez_y2 = coeffs.coeff_ez_y2().data(),
            .coeff_hx_1 = coeffs.coeff_hx_1().data(),
            .coeff_hx_2 = coeffs.coeff_hx_2().data(),
            .coeff_hy_1 = coeffs.coeff_hy_1().data(),
            .coeff_hy_2 = coeffs.coeff_hy_2().data(),
        };
    }

    // Run steps with the CPU block kernels over the whole grid.
    void run_blocks(Fields &fields, const Coeffs &coeffs, std::size_t pml, Boundary boundary)
    {
        const auto whole = fdtd::TMz::backends::serial::whole_block(fields.grid());
        for (int n = 0; n < STEPS; ++n) {
            fdtd::TMz::backends::serial::update_h_block(fields, coeffs, whole);
            fdtd::TMz::backends::serial::update_e_block(fields, coeffs, whole, pml, boundary);
        }
    }

    // Run steps with the cell functions, one cell at a time.
    void run_cells(Fields &fields, const Coeffs &coeffs, std::size_t pml, Boundary boundary)
    {
        const DeviceFields f = view(fields, pml);
        const DeviceCoeffs c = view(coeffs);
        for (int n = 0; n < STEPS; ++n) {
            for (int j = 0; j < f.ny; ++j) {
                for (int i = 0; i < f.nx; ++i) {
                    fdtd::TMz::backends::gpu::update_h_cell(f, c, i, j);
                }
            }
            for (int j = 0; j < f.ny; ++j) {
                for (int i = 0; i < f.nx; ++i) {
                    fdtd::TMz::backends::gpu::update_e_cell(f, c, i, j, boundary);
                }
            }
        }
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

    // Largest relative difference over the five fields.
    double field_difference(const Fields &a, const Fields &b)
    {
        return std::max({relative_difference(a.ez(), b.ez()), relative_difference(a.hx(), b.hx()),
                         relative_difference(a.hy(), b.hy()), relative_difference(a.ezx(), b.ezx()),
                         relative_difference(a.ezy(), b.ezy())});
    }

    // Check that two sets of fields hold the same bits.
    bool same(const Fields &a, const Fields &b)
    {
        return std::equal(a.ez().begin(), a.ez().end(), b.ez().begin()) &&
               std::equal(a.hx().begin(), a.hx().end(), b.hx().begin()) &&
               std::equal(a.hy().begin(), a.hy().end(), b.hy().begin()) &&
               std::equal(a.ezx().begin(), a.ezx().end(), b.ezx().begin()) &&
               std::equal(a.ezy().begin(), a.ezy().end(), b.ezy().begin());
    }
}

// Verify that the edge and interior tests split the cells as the CPU blocks do.
TEST_CASE("Cell classes match the CPU blocks", "[unit][cell]")
{
    const auto pml = GENERATE(std::size_t(0), std::size_t(1), std::size_t(3));
    const Grid grid(12, 9);
    Fields fields(grid);
    const DeviceFields f = view(fields, pml);
    const auto interior = fdtd::TMz::backends::serial::interior_block(grid, pml);
    for (int j = 0; j < f.ny; ++j) {
        for (int i = 0; i < f.nx; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            const auto uj = static_cast<std::size_t>(j);
            const bool edge = ui == 0 || uj == 0 || ui + 1 == grid.nx() || uj + 1 == grid.ny();
            const bool inside =
                ui >= interior.i0 && ui < interior.i1 && uj >= interior.j0 && uj < interior.j1;
            CHECK(fdtd::TMz::backends::gpu::is_edge_cell(f, i, j) == edge);
            CHECK(fdtd::TMz::backends::gpu::is_interior_cell(f, i, j) == inside);
        }
    }
}

// Verify that the cell functions advance the fields as the CPU block kernels do.
TEST_CASE("Cell updates match the CPU kernels", "[unit][cell]")
{
    const auto boundary = GENERATE(Boundary::Pec, Boundary::Pmc, Boundary::Pml);
    const auto size = GENERATE(Case{23, 19, 4}, Case{12, 31, 5}, Case{40, 40, 1}, Case{1, 1, 0},
                               Case{2, 5, 0}, Case{7, 1, 0});
    const std::size_t pml = boundary == Boundary::Pml ? size.pml : 0;
    const Grid grid(size.nx, size.ny, FloatType(1e-3), FloatType(2e-3));
    Material material(grid);
    material.set_pml(pml, 3, 1e-6);
    const Coeffs coeffs(material, fdtd::TMz::courant_dt(grid, 0.9));
    Fields blocks(grid);
    fill_distinct(blocks);
    Fields cells(grid);
    fill_distinct(cells);

    run_blocks(blocks, coeffs, pml, boundary);
    run_cells(cells, coeffs, pml, boundary);

    const double bound = TOLERANCE * double(std::numeric_limits<FloatType>::epsilon()) * STEPS;
    CHECK(field_difference(blocks, cells) <= bound);
    if (FDTD_EXACT) {
        CHECK(same(blocks, cells));
    }
}
