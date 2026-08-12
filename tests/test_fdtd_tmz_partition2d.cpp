/*
 * C++ file for the tests of the 2D TMz grid partitions.
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
#include "fdtd/TMz/backends/fdtd_tmz_partition2d.h"
#include "fdtd/TMz/fdtd_tmz_grid2d.h"

// Standard library for sizes, buffers and errors.
#include <cstddef>
#include <stdexcept>
#include <vector>

// Shorthand for the types under test.
using fdtd::TMz::Block2D;
using fdtd::TMz::Grid2D;
using fdtd::TMz::Partition2D;

// Helpers private to this file.
namespace
{
    // Check that every cell lies in exactly one block of one part, the part owner() names.
    void check_cover(const Partition2D &partition, const Grid2D &grid)
    {
        std::vector<int> count(grid.nx() * grid.ny(), 0);
        std::vector<std::size_t> part(grid.nx() * grid.ny(), 0);
        for (std::size_t p = 0; p < partition.size(); ++p) {
            for (const Block2D &block : partition.blocks(p)) {
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
TEST_CASE("Whole partition is the grid", "[unit][partition2d]")
{
    const Grid2D grid(7, 5);
    const Partition2D partition = Partition2D::whole(grid);
    REQUIRE(partition.size() == 1);
    REQUIRE(partition.blocks(0).size() == 1);
    CHECK(partition.blocks(0)[0].i1 == 7);
    CHECK(partition.blocks(0)[0].j1 == 5);
    CHECK(partition.tiles_x() == 1);
    CHECK(partition.tiles_y() == 1);
    check_cover(partition, grid);
}

// Verify that tiles cover the grid row-major, the last ones clipped.
TEST_CASE("Tiles cover the grid", "[unit][partition2d]")
{
    const auto w = GENERATE(std::size_t(1), std::size_t(4), std::size_t(7), std::size_t(50));
    const auto h = GENERATE(std::size_t(1), std::size_t(3), std::size_t(50));
    const Grid2D grid(23, 10);
    const Partition2D partition = Partition2D::tiles(grid, w, h);
    CHECK(partition.tiles_x() == (23 + w - 1) / w);
    CHECK(partition.tiles_y() == (10 + h - 1) / h);
    CHECK(partition.size() == partition.tiles_x() * partition.tiles_y());
    CHECK(partition.blocks(1 % partition.size())[0].i0 == (partition.tiles_x() > 1 ? w : 0));
    check_cover(partition, grid);
}

// Verify that ranges cover the grid in order with at most three blocks and even sizes.
TEST_CASE("Ranges cover the grid", "[unit][partition2d]")
{
    const auto parts =
        GENERATE(std::size_t(1), std::size_t(3), std::size_t(7), std::size_t(16), std::size_t(200));
    const Grid2D grid(13, 11);
    const Partition2D partition = Partition2D::ranges(grid, parts);
    REQUIRE(partition.size() == parts);
    CHECK(partition.tiles_x() == 0);
    std::size_t next = 0;
    for (std::size_t p = 0; p < parts; ++p) {
        CHECK(partition.blocks(p).size() <= 3);
        std::size_t cells = 0;
        for (const Block2D &block : partition.blocks(p)) {
            CHECK(block.j0 * 13 + block.i0 == next + cells);
            cells += (block.i1 - block.i0) * (block.j1 - block.j0);
        }
        CHECK(cells == (p + 1) * 143 / parts - p * 143 / parts);
        next += cells;
    }
    CHECK(next == 143);
    check_cover(partition, grid);
}

// Verify that empty tiles and no parts are rejected.
TEST_CASE("Partitions reject empty pieces", "[unit][partition2d]")
{
    const Grid2D grid(4, 4);
    CHECK_THROWS_AS(Partition2D::tiles(grid, 0, 2), std::invalid_argument);
    CHECK_THROWS_AS(Partition2D::tiles(grid, 2, 0), std::invalid_argument);
    CHECK_THROWS_AS(Partition2D::ranges(grid, 0), std::invalid_argument);
}
