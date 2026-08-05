/*
 * C++ file for the serial 2D TMz outer boundary.
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

// Function declaration.
#include "fdtd/TMz/serial/fdtd_tmz_serial_boundary.h"

// Standard library for sizes.
#include <cstddef>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Visit the edge cells once each: the first and last rows whole, then the first and
    // last cells of the rows between, which also covers grids one cell wide.
    template <typename Visit> static void for_each_edge(const Grid2D &grid, Visit visit)
    {
        const std::size_t nx = grid.nx();
        const std::size_t ny = grid.ny();
        for (std::size_t i = 0; i < nx; ++i) {
            visit(i, std::size_t(0));
            if (ny > 1) {
                visit(i, ny - 1);
            }
        }
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            visit(std::size_t(0), j);
            if (nx > 1) {
                visit(nx - 1, j);
            }
        }
    }

    // Zero the edge under PEC, or apply the Ez update with H outside the domain read as zero.
    void serial_update_boundary(Fields2D &fields, const Coeffs2D &coeffs,
                                Boundary boundary) noexcept
    {
        const Grid2D &grid = fields.grid();
        FloatType *__restrict Ez = fields.ez().data();

        if (boundary == Boundary::Pec) {
            for_each_edge(grid, [&](std::size_t i, std::size_t j) {
                Ez[grid.index(i, j)] = 0;
            });
            return;
        }

        const std::size_t nx = grid.nx();
        const std::size_t ny = grid.ny();
        const std::size_t pitch = grid.pitch();
        const FloatType invDx = FloatType(1) / grid.dx();
        const FloatType invDy = FloatType(1) / grid.dy();
        const FloatType *__restrict Hx = fields.hx().data();
        const FloatType *__restrict Hy = fields.hy().data();
        const FloatType *__restrict Ce1 = coeffs.ce1().data();
        const FloatType *__restrict Ce2 = coeffs.ce2().data();

        for_each_edge(grid, [&](std::size_t i, std::size_t j) {
            const std::size_t k = grid.index(i, j);
            const FloatType hyRight = i + 1 < nx ? Hy[k] : FloatType(0);
            const FloatType hyLeft = i > 0 ? Hy[k - 1] : FloatType(0);
            const FloatType hxTop = j + 1 < ny ? Hx[k] : FloatType(0);
            const FloatType hxBottom = j > 0 ? Hx[k - pitch] : FloatType(0);
            const FloatType curl = (hyRight - hyLeft) * invDx - (hxTop - hxBottom) * invDy;
            Ez[k] = Ce1[k] * Ez[k] + Ce2[k] * curl;
        });
    }
}
