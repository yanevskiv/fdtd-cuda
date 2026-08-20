/*
 * C++ file for the 2D TMz field array tests.
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
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/fdtd_aligned_buffer.h"

// Standard library for algorithms and integer pointers.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

// Check that every element of a view is zero.
static bool all_zero(std::span<const fdtd::FloatType> values)
{
    return std::all_of(values.begin(), values.end(), [](fdtd::FloatType v) {
        return v == 0;
    });
}

// Check that a pointer sits on the buffer alignment.
static bool is_aligned(const void *pointer)
{
    return reinterpret_cast<std::uintptr_t>(pointer) % fdtd::BUFFER_ALIGNMENT == 0;
}

// Verify that each array has the grid's size, is aligned and starts zeroed.
TEST_CASE("Fields allocates aligned zeroed arrays", "[unit][fields]")
{
    const auto [nx, ny] =
        GENERATE(table<std::size_t, std::size_t>({{1, 1}, {7, 5}, {64, 64}, {13, 31}}));
    const fdtd::TMz::Fields fields(fdtd::TMz::Grid(nx, ny));

    CHECK(fields.grid().nx() == nx);
    CHECK(fields.grid().ny() == ny);
    for (auto view : {fields.ez(), fields.hx(), fields.hy(), fields.ezx(), fields.ezy()}) {
        CHECK(view.size() == fields.grid().size());
        CHECK(is_aligned(view.data()));
        CHECK(all_zero(view));
    }
    CHECK(fields.ez().data() != fields.hx().data());
    CHECK(fields.hx().data() != fields.hy().data());
    CHECK(fields.hy().data() != fields.ezx().data());
    CHECK(fields.ezx().data() != fields.ezy().data());
}

// Verify that zero() clears values written through the views.
TEST_CASE("Fields zero clears every field", "[unit][fields]")
{
    fdtd::TMz::Fields fields(fdtd::TMz::Grid(9, 4));
    const auto &grid = fields.grid();

    fields.ez()[grid.index(3, 2)] = 1;
    fields.hx()[grid.index(0, 0)] = 2;
    fields.hy()[grid.index(8, 3)] = 3;
    fields.ezx()[grid.index(1, 1)] = 4;
    fields.ezy()[grid.index(7, 2)] = 5;
    CHECK_FALSE(all_zero(fields.ez()));
    CHECK_FALSE(all_zero(fields.ezx()));
    CHECK_FALSE(all_zero(fields.ezy()));
    fields.zero();
    CHECK(all_zero(fields.ez()));
    CHECK(all_zero(fields.hx()));
    CHECK(all_zero(fields.hy()));
    CHECK(all_zero(fields.ezx()));
    CHECK(all_zero(fields.ezy()));
}

// Verify that moving a buffer transfers its memory and leaves the source empty.
TEST_CASE("AlignedBuffer moves its memory", "[unit][aligned_buffer]")
{
    fdtd::AlignedBuffer<fdtd::FloatType> source(10);
    source[4] = 5;
    const auto *memory = source.data();

    fdtd::AlignedBuffer<fdtd::FloatType> target(std::move(source));
    CHECK(target.data() == memory);
    CHECK(target.size() == 10);
    CHECK(target[4] == fdtd::FloatType{5});
    CHECK(source.data() == nullptr);
    CHECK(source.size() == 0);

    fdtd::AlignedBuffer<fdtd::FloatType> empty;
    CHECK(empty.data() == nullptr);
    CHECK(empty.size() == 0);
}
