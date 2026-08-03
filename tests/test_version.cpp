/*
 * C++ file for the version and floating-point type tests.
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
#include "fdtd/fdtd.h"

// Standard library for type comparisons.
#include <string>
#include <type_traits>

// Verify that the header version matches the version CMake was given.
TEST_CASE("Header version matches the CMake project version", "[unit][version]")
{
    CHECK(FDTD_VERSION_MAJOR == FDTD_TEST_PROJECT_VERSION_MAJOR);
    CHECK(FDTD_VERSION_MINOR == FDTD_TEST_PROJECT_VERSION_MINOR);
    CHECK(FDTD_VERSION_PATCH == FDTD_TEST_PROJECT_VERSION_PATCH);
    CHECK(std::string(FDTD_VERSION_STRING) == FDTD_TEST_PROJECT_VERSION);
}

// Verify that fdtd::Real follows the FDTD_DOUBLE option.
TEST_CASE("Real follows FDTD_DOUBLE", "[unit][real]")
{
    using Expected = std::conditional_t<FDTD_DOUBLE != 0, double, float>;
    STATIC_CHECK(std::is_same_v<fdtd::Real, Expected>);
}
