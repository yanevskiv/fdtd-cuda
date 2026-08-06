/*
 * C++ tests for source waveforms.
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

// Project header under test.
#include "fdtd/fdtd_waveform.h"

// Standard library for math, limits and errors.
#include <cmath>
#include <limits>
#include <stdexcept>

// Shorthand for the types under test.
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fdtd::FloatType;
using fdtd::Gaussian;
using fdtd::sample_waveform;

// Verify the peak, the symmetry about t0 and the value one width away.
TEST_CASE("Gaussian shape", "[unit][waveform]")
{
    const Gaussian g(4e-12, 1e-12, 2.0);
    CHECK_THAT(double(g(FloatType(4e-12))), WithinRel(2.0, 1e-6));
    CHECK_THAT(double(g(FloatType(5e-12))), WithinRel(2.0 * std::exp(-1.0), 1e-5));
    CHECK_THAT(double(g(FloatType(3e-12))), WithinRel(double(g(FloatType(5e-12))), 1e-5));
}

// Verify that from_steps peaks three widths in and starts below exp(-9).
TEST_CASE("Gaussian from_steps", "[unit][waveform]")
{
    const double dt = 1e-12;
    const Gaussian g = Gaussian::from_steps(dt, 20);
    CHECK_THAT(g.tau(), WithinRel(20 * dt, 1e-12));
    CHECK_THAT(g.t0(), WithinRel(60 * dt, 1e-12));
    CHECK(g.amplitude() == 1);
    CHECK(double(g(FloatType(0))) <= std::exp(-9.0) * (1 + 1e-6));
}

// Verify that bad parameters throw.
TEST_CASE("Gaussian rejects bad parameters", "[unit][waveform]")
{
    const double inf = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(Gaussian(0, 0), std::invalid_argument);
    CHECK_THROWS_AS(Gaussian(0, -1e-12), std::invalid_argument);
    CHECK_THROWS_AS(Gaussian(0, inf), std::invalid_argument);
    CHECK_THROWS_AS(Gaussian(inf, 1e-12), std::invalid_argument);
    CHECK_THROWS_AS(Gaussian(0, 1e-12, inf), std::invalid_argument);
    CHECK_THROWS_AS(Gaussian::from_steps(0), std::invalid_argument);
    CHECK_THROWS_AS(Gaussian::from_steps(1e-12, 0), std::invalid_argument);
}

// Verify one sample per step at t_first + n * dt, including a half-step offset.
TEST_CASE("sample_waveform takes one value per step", "[unit][waveform]")
{
    const double dt = 1e-12;
    const Gaussian g = Gaussian::from_steps(dt, 10);
    const auto samples = sample_waveform(g, dt, 0, 100, 0.5);
    REQUIRE(samples.size() == 100);
    for (std::size_t n = 0; n < samples.size(); ++n) {
        CHECK(samples[n] == g(FloatType((double(n) + 0.5) * dt)));
    }
    CHECK_THAT(double(samples[30]), WithinAbs(1.0, 1e-2));
}

// Verify that a lambda works as a waveform and an empty one throws.
TEST_CASE("sample_waveform takes any callable", "[unit][waveform]")
{
    const auto samples = sample_waveform(
        [](FloatType t) {
            return t;
        },
        0.5, 2, 3);
    CHECK(samples == std::vector<FloatType>{FloatType(1), FloatType(1.5), FloatType(2)});
    CHECK(sample_waveform(
              [](FloatType) {
                  return FloatType(0);
              },
              1, 0, 0)
              .empty());
    CHECK_THROWS_AS(sample_waveform(fdtd::Waveform{}, 1, 0, 1), std::invalid_argument);
}

// Verify that sampling in two calls gives the same values as one call, bit for bit.
TEST_CASE("sample_waveform splits without drift", "[unit][waveform]")
{
    const double dt = 1.3e-12;
    const Gaussian g = Gaussian::from_steps(dt, 10);
    const auto whole = sample_waveform(g, dt, 0, 100, 0.5);
    auto split = sample_waveform(g, dt, 0, 37, 0.5);
    const auto rest = sample_waveform(g, dt, 37, 63, 0.5);
    split.insert(split.end(), rest.begin(), rest.end());
    CHECK(split == whole);
}
