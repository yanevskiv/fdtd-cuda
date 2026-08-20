/*
 * C++ file for the tests of the CUDA backend's coefficient table.
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

// Project headers under test.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_table.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_material.h"

// Standard library for sizes and sets.
#include <cstddef>
#include <cstdint>
#include <set>

// Shorthand for the types under test.
using fdtd::FloatType;
using fdtd::TMz::Coeffs;
using fdtd::TMz::courant_dt;
using fdtd::TMz::Grid;
using fdtd::TMz::Material;
using fdtd::TMz::backends::gpu::CoeffTable;
using fdtd::TMz::backends::gpu::CoeffTuple;

// Helpers private to this file.
namespace
{
    // Check that a tuple holds a cell's values exactly.
    void check_tuple(const CoeffTuple &tuple, const Coeffs &coeffs, std::size_t k)
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

    // Check every cell: an entry holds its values, and a cell reads the arrays only when its Ez
    // pairs differ or the table is full.
    void check_cells(const CoeffTable &table, const Coeffs &coeffs)
    {
        const auto index = table.index();
        REQUIRE(index.size() == coeffs.grid().size());
        REQUIRE(table.entries().size() <= CoeffTable::CAPACITY);
        for (std::size_t k = 0; k < index.size(); ++k) {
            INFO("cell " << k);
            const bool same = coeffs.coeff_ez_x1()[k] == coeffs.coeff_ez_y1()[k] &&
                              coeffs.coeff_ez_x2()[k] == coeffs.coeff_ez_y2()[k];
            if (index[k] == CoeffTable::ARRAYS) {
                CHECK((!same || table.entries().size() == CoeffTable::CAPACITY));
            } else {
                REQUIRE(index[k] < table.entries().size());
                check_tuple(table.entries()[index[k]], coeffs, k);
            }
        }
    }
}

// Verify that vacuum without a PML is one entry that every cell takes.
TEST_CASE("Coefficient table of vacuum", "[unit][cuda][coeffs]")
{
    const Grid grid(9, 7, 1e-3F, 2e-3F);
    const Material material(grid);
    const Coeffs coeffs(material, courant_dt(grid, 0.9));
    const CoeffTable table(coeffs);
    REQUIRE(table.entries().size() == 1);
    for (const std::uint8_t entry : table.index()) {
        CHECK(entry == 0);
    }
    check_cells(table, coeffs);
    REQUIRE(table.uniform_entry(1) != nullptr);
    CHECK(table.uniform_entry(1) == &table.entries()[0]);
    CHECK(table.uniform_entry(2) != nullptr);
    CHECK(table.uniform_entry(3) == nullptr);
    CHECK(table.uniform_entry(4) == nullptr);
    CHECK(table.grid().nx() == 9);
}

// Verify that the PML cells read the arrays and the core inside the PML shares one entry, while
// the interior's last column holds Hy half a cell into the PML.
TEST_CASE("Coefficient table of vacuum with a PML", "[unit][cuda][coeffs]")
{
    const Grid grid(30, 24);
    Material material(grid);
    material.set_pml(4, 3, 1e-6);
    const Coeffs coeffs(material, courant_dt(grid, 0.9));
    const CoeffTable table(coeffs);
    check_cells(table, coeffs);
    CHECK(table.index()[grid.index(0, 12)] == CoeffTable::ARRAYS);
    CHECK(table.index()[grid.index(15, 2)] == CoeffTable::ARRAYS);
    const CoeffTuple *uniform = table.uniform_entry(4);
    REQUIRE(uniform != nullptr);
    check_tuple(*uniform, coeffs, grid.index(15, 12));
    CHECK(table.index()[grid.index(25, 12)] != table.index()[grid.index(15, 12)]);
    CHECK(table.uniform_entry(1) == nullptr);
}

// Verify that media give an entry each and no rectangle that spans two of them is uniform.
TEST_CASE("Coefficient table of a slab and a lossy circle", "[unit][cuda][coeffs]")
{
    const Grid grid(40, 32);
    Material material(grid);
    material.fill_rect(10, 0, 16, 32, {.eps_r = 4});
    material.fill_circle(28, 16, 5, {.eps_r = 2, .sigma_e = FloatType(0.5)});
    const Coeffs coeffs(material, courant_dt(grid, 0.9));
    const CoeffTable table(coeffs);
    check_cells(table, coeffs);
    CHECK(table.entries().size() >= 3);
    CHECK(table.index()[grid.index(12, 5)] != table.index()[grid.index(2, 5)]);
    CHECK(table.index()[grid.index(28, 16)] != table.index()[grid.index(2, 5)]);
    CHECK(table.index()[grid.index(28, 16)] != table.index()[grid.index(12, 5)]);
    CHECK(table.uniform_entry(1) == nullptr);
}

// Verify that a grid with more distinct tuples than the table holds fills the table, and the
// cells past it read the arrays: cells 0 to 299 in row-major order take a medium each, cell 0's
// being vacuum, so cells 255 to 299 find no room and the vacuum after them takes entry 0.
TEST_CASE("Coefficient table past its capacity", "[unit][cuda][coeffs]")
{
    const Grid grid(20, 20);
    Material material(grid);
    for (std::size_t k = 0; k < 300; ++k) {
        const std::size_t i = k % 20;
        const std::size_t j = k / 20;
        material.fill_rect(i, j, i + 1, j + 1, {.eps_r = FloatType(1) + FloatType(k) / 64});
    }
    const Coeffs coeffs(material, courant_dt(grid, 0.9));
    const CoeffTable table(coeffs);
    CHECK(table.entries().size() == CoeffTable::CAPACITY);
    check_cells(table, coeffs);
    std::set<std::uint8_t> used(table.index().begin(), table.index().end());
    CHECK(used.contains(CoeffTable::ARRAYS));
    CHECK(used.size() == CoeffTable::CAPACITY + 1);
    CHECK(table.index()[grid.index(19, 14)] == CoeffTable::ARRAYS);
    CHECK(table.index()[grid.index(19, 19)] == 0);
}
