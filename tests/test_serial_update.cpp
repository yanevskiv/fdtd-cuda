/*
 * C++ file for the serial 2D TMz Yee update tests.
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
#include "fdtd/TMz/fdtd_tmz_courant.h"
#include "fdtd/TMz/serial/fdtd_tmz_serial_update.h"

// Standard library for sizes.
#include <cstddef>

// Shorthand for the types under test.
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::TMz::Coeffs2D;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Fields2D;
using fdtd::TMz::Grid2D;
using fdtd::TMz::Material2D;

// Relative tolerance that holds in single precision.
static constexpr double TOL = 1e-5;

// Verify one step from a unit Ez impulse in vacuum on square cells with S = 1/2.
// Then c^2 dt^2 / dx^2 = S^2 / 2 = 1/8, so the centre falls to 1 - 4/8 = 1/2 and
// each of its four neighbours rises to 1/8.
TEST_CASE("Serial update of an Ez impulse", "[unit][serial_update]")
{
    const Grid2D grid(7, 7, 1e-3F, 1e-3F);
    const Material2D material(grid);
    const Coeffs2D coeffs(material, courant_dt(grid, 0.5));
    Fields2D fields(grid);
    const std::size_t c = grid.index(3, 3);
    fields.ez()[c] = 1;

    fdtd::TMz::serial_update_h(fields, coeffs);
    const double h = double(coeffs.ch2()[c]) / double(grid.dx());
    CHECK_THAT(double(fields.hy()[grid.index(2, 3)]), WithinRel(h, TOL));
    CHECK_THAT(double(fields.hy()[grid.index(3, 3)]), WithinRel(-h, TOL));
    CHECK_THAT(double(fields.hx()[grid.index(3, 2)]), WithinRel(-h, TOL));
    CHECK_THAT(double(fields.hx()[grid.index(3, 3)]), WithinRel(h, TOL));
    CHECK(fields.hx()[grid.index(2, 3)] == 0);
    CHECK(fields.hy()[grid.index(3, 2)] == 0);

    fdtd::TMz::serial_update_e(fields, coeffs);
    CHECK_THAT(double(fields.ez()[c]), WithinRel(0.5, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(2, 3)]), WithinRel(0.125, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(4, 3)]), WithinRel(0.125, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(3, 2)]), WithinRel(0.125, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(3, 4)]), WithinRel(0.125, TOL));
    CHECK_THAT(double(fields.ez()[grid.index(2, 2)]), WithinAbs(0.0, 1e-12));
}

// Verify that the update leaves the edge Ez, the last-row Hx and the last-column
// Hy untouched.
TEST_CASE("Serial update leaves the edges alone", "[unit][serial_update]")
{
    const Grid2D grid(4, 3);
    const Material2D material(grid);
    const Coeffs2D coeffs(material, courant_dt(grid, 0.9));
    Fields2D fields(grid);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            fields.ez()[grid.index(i, j)] = FloatType(1 + i + 10 * j);
        }
    }
    const std::size_t lastRow = grid.ny() - 1;
    const std::size_t lastCol = grid.nx() - 1;

    fdtd::TMz::serial_update_h(fields, coeffs);
    fdtd::TMz::serial_update_e(fields, coeffs);

    for (std::size_t i = 0; i < grid.nx(); ++i) {
        CHECK(fields.hx()[grid.index(i, lastRow)] == 0);
        CHECK(fields.ez()[grid.index(i, 0)] == FloatType(1 + i));
        CHECK(fields.ez()[grid.index(i, lastRow)] == FloatType(1 + i + 10 * lastRow));
    }
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        CHECK(fields.hy()[grid.index(lastCol, j)] == 0);
        CHECK(fields.ez()[grid.index(0, j)] == FloatType(1 + 10 * j));
        CHECK(fields.ez()[grid.index(lastCol, j)] == FloatType(1 + lastCol + 10 * j));
    }
}

// Verify that one-cell-wide grids update without touching memory outside them.
TEST_CASE("Serial update on degenerate grids", "[unit][serial_update]")
{
    for (const Grid2D &grid : {Grid2D(1, 1), Grid2D(1, 5), Grid2D(5, 1), Grid2D(2, 2)}) {
        const Material2D material(grid);
        const Coeffs2D coeffs(material, courant_dt(grid, 0.9));
        Fields2D fields(grid);
        fields.ez()[0] = 1;

        fdtd::TMz::serial_update_h(fields, coeffs);
        fdtd::TMz::serial_update_e(fields, coeffs);

        CHECK(fields.ez()[0] == 1);
    }
}
