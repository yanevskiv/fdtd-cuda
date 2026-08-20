/*
 * C++ file for the 2D TMz OpenMP backend's level v3, table.
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

// Level declarations.
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_levels.h"

// The parallel region and the table kernels.
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_team.h"
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_v3.h"

// Standard library for bit patterns, sizes and the optional tuple.
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

// Namespace for the levels of the 2D TMz OpenMP backend.
namespace fdtd::TMz::backends::omp
{
    // Helpers private to this file.
    namespace
    {
        // Unsigned integer as wide as FloatType, for comparing values bit by bit.
        using Bits = std::conditional_t<sizeof(FloatType) == 8, std::uint64_t, std::uint32_t>;

        // Whether two values hold the same bits.
        bool same_bits(FloatType a, FloatType b) noexcept
        {
            return std::bit_cast<Bits>(a) == std::bit_cast<Bits>(b);
        }
    }

    // Take the core's first cell, then check that every cell of the core holds its bits.
    std::optional<CoeffTuple> core_tuple(const Coeffs &coeffs, std::size_t lo)
    {
        const Grid &grid = coeffs.grid();
        if (grid.nx() <= 2 * lo + 1 || grid.ny() <= 2 * lo + 1) {
            return std::nullopt;
        }
        const auto ezX1 = coeffs.coeff_ez_x1();
        const auto ezX2 = coeffs.coeff_ez_x2();
        const auto ezY1 = coeffs.coeff_ez_y1();
        const auto ezY2 = coeffs.coeff_ez_y2();
        const auto hx1 = coeffs.coeff_hx_1();
        const auto hx2 = coeffs.coeff_hx_2();
        const auto hy1 = coeffs.coeff_hy_1();
        const auto hy2 = coeffs.coeff_hy_2();
        const std::size_t first = grid.index(lo, lo);
        const CoeffTuple tuple{
            .ez_1 = ezX1[first],
            .ez_2 = ezX2[first],
            .hx_1 = hx1[first],
            .hx_2 = hx2[first],
            .hy_1 = hy1[first],
            .hy_2 = hy2[first],
        };
        for (std::size_t j = lo; j < grid.ny() - lo - 1; ++j) {
            for (std::size_t i = lo; i < grid.nx() - lo - 1; ++i) {
                const std::size_t k = grid.index(i, j);
                if (!same_bits(ezX1[k], tuple.ez_1) || !same_bits(ezY1[k], tuple.ez_1) ||
                    !same_bits(ezX2[k], tuple.ez_2) || !same_bits(ezY2[k], tuple.ez_2) ||
                    !same_bits(hx1[k], tuple.hx_1) || !same_bits(hx2[k], tuple.hx_2) ||
                    !same_bits(hy1[k], tuple.hy_1) || !same_bits(hy2[k], tuple.hy_2)) {
                    return std::nullopt;
                }
            }
        }
        return tuple;
    }

    // Share the rows of each phase among the threads, as v2, the core reading the tuple.
    void run_v3(const Advance &advance, const std::optional<SharedTuple> &shared)
    {
        const serial::Stepper &stepper = advance.stepper;
        const std::size_t rows = stepper.partition().size();
        const SharedTuple *tuple = shared.has_value() ? &*shared : nullptr;
        run_team(advance.threads, advance.bind, [&] {
            for (std::size_t n = 0; n < advance.nsteps; ++n) {
#pragma omp for schedule(static)
                for (std::size_t j = 0; j < rows; ++j) {
                    v3::h_phase(stepper, tuple, j, n);
                }
#pragma omp for schedule(static)
                for (std::size_t j = 0; j < rows; ++j) {
                    v3::e_phase(stepper, tuple, j, n);
                }
            }
        });
    }
}
