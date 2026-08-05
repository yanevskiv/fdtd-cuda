/*
 * C++ file for the serial 2D TMz Yee update.
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

// Function declarations.
#include "fdtd/TMz/serial/fdtd_tmz_serial_update.h"

// Standard library for sizes.
#include <cstddef>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Update Hx row by row below the last row, then Hy row by row left of the last column.
    void serial_update_h(Fields2D &fields, const Coeffs2D &coeffs) noexcept
    {
        const Grid2D &grid = fields.grid();
        const std::size_t nx = grid.nx();
        const std::size_t ny = grid.ny();
        const std::size_t pitch = grid.pitch();
        const FloatType invDx = FloatType(1) / grid.dx();
        const FloatType invDy = FloatType(1) / grid.dy();

        const FloatType *__restrict Ez = fields.ez().data();
        FloatType *__restrict Hx = fields.hx().data();
        FloatType *__restrict Hy = fields.hy().data();
        const FloatType *__restrict Ch1 = coeffs.ch1().data();
        const FloatType *__restrict Ch2 = coeffs.ch2().data();

        for (std::size_t j = 0; j + 1 < ny; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t k = row + i;
                Hx[k] = Ch1[k] * Hx[k] - Ch2[k] * invDy * (Ez[k + pitch] - Ez[k]);
            }
        }

        for (std::size_t j = 0; j < ny; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = 0; i + 1 < nx; ++i) {
                const std::size_t k = row + i;
                Hy[k] = Ch1[k] * Hy[k] + Ch2[k] * invDx * (Ez[k + 1] - Ez[k]);
            }
        }
    }

    // Update the interior Ez row by row.
    void serial_update_e(Fields2D &fields, const Coeffs2D &coeffs) noexcept
    {
        const Grid2D &grid = fields.grid();
        const std::size_t nx = grid.nx();
        const std::size_t ny = grid.ny();
        const std::size_t pitch = grid.pitch();
        const FloatType invDx = FloatType(1) / grid.dx();
        const FloatType invDy = FloatType(1) / grid.dy();

        FloatType *__restrict Ez = fields.ez().data();
        const FloatType *__restrict Hx = fields.hx().data();
        const FloatType *__restrict Hy = fields.hy().data();
        const FloatType *__restrict Ce1 = coeffs.ce1().data();
        const FloatType *__restrict Ce2 = coeffs.ce2().data();

        for (std::size_t j = 1; j + 1 < ny; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = 1; i + 1 < nx; ++i) {
                const std::size_t k = row + i;
                const FloatType curl =
                    (Hy[k] - Hy[k - 1]) * invDx - (Hx[k] - Hx[k - pitch]) * invDy;
                Ez[k] = Ce1[k] * Ez[k] + Ce2[k] * curl;
            }
        }
    }
}
