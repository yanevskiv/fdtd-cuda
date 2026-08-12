/*
 * C++ file for the 2D TMz update kernels over a block of cells.
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

// Struct and kernel declarations.
#include "fdtd/TMz/backends/fdtd_tmz_block2d.h"

// Standard library for bounds and sizes.
#include <algorithm>
#include <cstddef>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Helpers private to this file.
    namespace
    {
        // Visit each edge cell of a block once: the first and last rows, then the row ends between.
        template <typename Visit>
        void for_each_edge(const Grid2D &grid, const Block2D &block, Visit visit)
        {
            const std::size_t nx = grid.nx();
            const std::size_t ny = grid.ny();
            if (block.empty()) {
                return;
            }
            for (std::size_t i = block.i0; i < block.i1; ++i) {
                if (block.j0 == 0) {
                    visit(i, std::size_t(0));
                }
                if (ny > 1 && block.j1 == ny) {
                    visit(i, ny - 1);
                }
            }
            const std::size_t jBegin = std::max<std::size_t>(block.j0, 1);
            const std::size_t jEnd = std::min(block.j1, ny - 1);
            for (std::size_t j = jBegin; j < jEnd; ++j) {
                if (block.i0 == 0) {
                    visit(std::size_t(0), j);
                }
                if (nx > 1 && block.i1 == nx) {
                    visit(nx - 1, j);
                }
            }
        }
    }

    // Take the arrays of a coefficient set.
    CoeffsView::CoeffsView(const Coeffs2D &coeffs) noexcept
        : CoeffsView(coeffs.coeff_ez_x1().data(), coeffs.coeff_ez_x2().data(),
                     coeffs.coeff_ez_y1().data(), coeffs.coeff_ez_y2().data(),
                     coeffs.coeff_hx_1().data(), coeffs.coeff_hx_2().data(),
                     coeffs.coeff_hy_1().data(), coeffs.coeff_hy_2().data())
    {
        // Empty.
    }

    // Take eight arrays.
    CoeffsView::CoeffsView(const FloatType *ezX1, const FloatType *ezX2, const FloatType *ezY1,
                           const FloatType *ezY2, const FloatType *hx1, const FloatType *hx2,
                           const FloatType *hy1, const FloatType *hy2) noexcept
        : coeff_ez_x1(ezX1)
        , coeff_ez_x2(ezX2)
        , coeff_ez_y1(ezY1)
        , coeff_ez_y2(ezY2)
        , coeff_hx_1(hx1)
        , coeff_hx_2(hx2)
        , coeff_hy_1(hy1)
        , coeff_hy_2(hy2)
    {
        // Empty.
    }

    // Zero the block's edge Ez and its parts, or update it reading H outside the domain as zero.
    void update_boundary_block(Fields2D &fields, CoeffsView coeffs, const Block2D &block,
                               Boundary boundary) noexcept
    {
        const Grid2D &grid = fields.grid();
        FloatType *__restrict Ez = fields.ez().data();

        if (boundary != Boundary::Pmc) {
            FloatType *__restrict Ezx = fields.ezx().data();
            FloatType *__restrict Ezy = fields.ezy().data();
            for_each_edge(grid, block, [&](std::size_t i, std::size_t j) {
                const std::size_t k = grid.index(i, j);
                Ez[k] = 0;
                Ezx[k] = 0;
                Ezy[k] = 0;
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
        const FloatType *__restrict Cex1 = coeffs.coeff_ez_x1;
        const FloatType *__restrict Cex2 = coeffs.coeff_ez_x2;

        for_each_edge(grid, block, [&](std::size_t i, std::size_t j) {
            const std::size_t k = grid.index(i, j);
            const FloatType hyRight = i + 1 < nx ? Hy[k] : FloatType(0);
            const FloatType hyLeft = i > 0 ? Hy[k - 1] : FloatType(0);
            const FloatType hxTop = j + 1 < ny ? Hx[k] : FloatType(0);
            const FloatType hxBottom = j > 0 ? Hx[k - pitch] : FloatType(0);
            const FloatType curl = (hyRight - hyLeft) * invDx - (hxTop - hxBottom) * invDy;
            Ez[k] = Cex1[k] * Ez[k] + Cex2[k] * curl;
        });
    }
}
