/*
 * C++ file for the serial 2D TMz backend routine tests: the Yee update and the outer boundary.
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
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers under test.
#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"

// Standard library for math, sizes and buffers.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Shorthand for the types under test.
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::TMz::Boundary;
using fdtd::TMz::Coeffs;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Fields;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;

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

// Run whole steps: H, interior E, then the boundary.
static void run(Fields &fields, const Coeffs &coeffs, Boundary boundary, int nsteps)
{
    for (int n = 0; n < nsteps; ++n) {
        fdtd::TMz::serial_update_h(fields, coeffs);
        fdtd::TMz::serial_update_e(fields, coeffs, 0);
        fdtd::TMz::serial_update_boundary(fields, coeffs, boundary);
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

    fdtd::TMz::serial_update_h(fields, coeffs);
    const double h = double(coeffs.coeff_hy_2()[c]) / double(grid.dx());
    CHECK_THAT(double(fields.hy()[grid.index(2, 3)]), WithinRel(h, TOL));
    CHECK_THAT(double(fields.hy()[grid.index(3, 3)]), WithinRel(-h, TOL));
    CHECK_THAT(double(fields.hx()[grid.index(3, 2)]), WithinRel(-h, TOL));
    CHECK_THAT(double(fields.hx()[grid.index(3, 3)]), WithinRel(h, TOL));
    CHECK(fields.hx()[grid.index(2, 3)] == FloatType(0));
    CHECK(fields.hy()[grid.index(3, 2)] == FloatType(0));

    fdtd::TMz::serial_update_e(fields, coeffs, 0);
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
    const fdtd::TMz::CoeffsView coeffs(ezX.data(), zero.data(), ezY.data(), zero.data(), hx.data(),
                                       zero.data(), hy.data(), zero.data());
    Fields fields(grid);
    for (std::size_t k = 0; k < grid.size(); ++k) {
        fields.ez()[k] = 1;
        fields.hx()[k] = 1;
        fields.hy()[k] = 1;
    }

    fdtd::TMz::serial_update_h(fields, coeffs);
    fdtd::TMz::serial_update_e(fields, coeffs, 0);

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

    fdtd::TMz::serial_update_h(fields, coeffs);
    fdtd::TMz::serial_update_e(fields, coeffs, 0);

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

        fdtd::TMz::serial_update_h(fields, coeffs);
        fdtd::TMz::serial_update_e(fields, coeffs, 0);

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

    fdtd::TMz::serial_update_boundary(fields, coeffs, Boundary::Pec);

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

    fdtd::TMz::serial_update_boundary(fields, coeffs, Boundary::Pmc);

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

        fdtd::TMz::serial_update_boundary(fields, coeffs, Boundary::Pmc);
        fdtd::TMz::serial_update_boundary(fields, coeffs, Boundary::Pec);

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

    fdtd::TMz::serial_update_e(plain, coeffs, 0);
    fdtd::TMz::serial_update_e(split, coeffs, pml);
    fdtd::TMz::serial_update_pml(split, coeffs, pml);

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

    fdtd::TMz::serial_update_pml(fields, coeffs, 4);

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

    fdtd::TMz::serial_update_boundary(fields, coeffs, Boundary::Pml);

    CHECK(fields.ezx()[grid.index(0, 2)] == FloatType(0));
    CHECK(fields.ezy()[grid.index(5, 4)] == FloatType(0));
    CHECK(fields.ezx()[grid.index(3, 0)] == FloatType(0));
    CHECK(fields.ezy()[grid.index(2, 2)] == FloatType(2));
}
