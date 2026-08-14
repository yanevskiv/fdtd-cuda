/*
 * C++ header for the 2D TMz update kernels over a block of cells.
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

#ifndef FDTD_TMZ_BACKEND_CPU_BLOCK_H
#define FDTD_TMZ_BACKEND_CPU_BLOCK_H

#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/fdtd_float.h"

#include <algorithm>
#include <cstddef>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /** Namespace for what the 2D TMz CPU backends share. */
            namespace cpu
            {
                /**
                 * Pointers to the eight coefficient arrays the update kernels read.
                 *
                 * The kernels take these rather than a Coeffs so a backend can keep its own
                 * arrays, as a GPU backend keeps device arrays. A Coeffs converts implicitly.
                 */
                struct CoeffsView
                {
                    /**
                     * View the arrays of a coefficient set, which must outlive the view.
                     *
                     * @param coeffs Coefficients to view.
                     */
                    CoeffsView(
                        const Coeffs &coeffs) noexcept; // NOLINT(google-explicit-constructor)

                    /**
                     * View eight arrays over the same grid, laid out as in Coeffs.
                     *
                     * @param ezX1 Ez decay factors along x.
                     * @param ezX2 Ez curl factors along x.
                     * @param ezY1 Ez decay factors along y.
                     * @param ezY2 Ez curl factors along y.
                     * @param hx1 Hx decay factors.
                     * @param hx2 Hx curl factors.
                     * @param hy1 Hy decay factors.
                     * @param hy2 Hy curl factors.
                     */
                    CoeffsView(const FloatType *ezX1, const FloatType *ezX2, const FloatType *ezY1,
                               const FloatType *ezY2, const FloatType *hx1, const FloatType *hx2,
                               const FloatType *hy1, const FloatType *hy2) noexcept;

                    /** Ez decay factors along x. */
                    const FloatType *coeff_ez_x1;

                    /** Ez curl factors along x, before 1/dx. */
                    const FloatType *coeff_ez_x2;

                    /** Ez decay factors along y. */
                    const FloatType *coeff_ez_y1;

                    /** Ez curl factors along y, before 1/dy. */
                    const FloatType *coeff_ez_y2;

                    /** Hx decay factors. */
                    const FloatType *coeff_hx_1;

                    /** Hx curl factors, before 1/dy. */
                    const FloatType *coeff_hx_2;

                    /** Hy decay factors. */
                    const FloatType *coeff_hy_1;

                    /** Hy curl factors, before 1/dx. */
                    const FloatType *coeff_hy_2;
                };

                /** A half-open rectangle of cells, i0 <= i < i1 and j0 <= j < j1. */
                struct Block
                {
                    /** First column. */
                    std::size_t i0;

                    /** One past the last column. */
                    std::size_t i1;

                    /** First row. */
                    std::size_t j0;

                    /** One past the last row. */
                    std::size_t j1;

                    /**
                     * Whether the block holds no cell.
                     *
                     * @return True when either side is empty.
                     */
                    inline bool empty() const noexcept
                    {
                        return i0 >= i1 || j0 >= j1;
                    }
                };

                /**
                 * Block of every cell of a grid.
                 *
                 * @param grid Grid to cover.
                 * @return The block from (0, 0) to (nx, ny).
                 */
                inline Block whole_block(const Grid &grid) noexcept
                {
                    return {
                        .i0 = 0,
                        .i1 = grid.nx(),
                        .j0 = 0,
                        .j1 = grid.ny(),
                    };
                }

                /**
                 * Cells two blocks share.
                 *
                 * @param a First block.
                 * @param b Second block.
                 * @return The intersection, empty() when they share none.
                 */
                inline Block intersect(const Block &a, const Block &b) noexcept
                {
                    return {
                        .i0 = std::max(a.i0, b.i0),
                        .i1 = std::min(a.i1, b.i1),
                        .j0 = std::max(a.j0, b.j0),
                        .j1 = std::min(a.j1, b.j1),
                    };
                }

                /**
                 * Call a body for every index of a range, as one `omp simd` loop when asked and
                 * OpenMP is enabled.
                 *
                 * @tparam Simd Whether the loop carries `omp simd`.
                 * @tparam Body Callable taking the index.
                 * @param begin First index.
                 * @param end One past the last index.
                 * @param body Body of the loop, free of dependences between indices.
                 */
                template <bool Simd, typename Body>
                inline void for_range(std::size_t begin, std::size_t end, Body body) noexcept
                {
                    if constexpr (Simd) {
#ifdef _OPENMP
#pragma omp simd
#endif
                        for (std::size_t i = begin; i < end; ++i) {
                            body(i);
                        }
                    } else {
                        for (std::size_t i = begin; i < end; ++i) {
                            body(i);
                        }
                    }
                }

                /**
                 * Advance Hx and Hy over a block by one step from the differences of Ez.
                 *
                 * With Hx(i, j) at (i, j + 1/2) and Hy(i, j) at (i + 1/2, j):
                 *
                 *     Hx(i, j) = Chx1 Hx(i, j) - Chx2 / dy (Ez(i, j + 1) - Ez(i, j)),  j < ny - 1,
                 *     Hy(i, j) = Chy1 Hy(i, j) + Chy2 / dx (Ez(i + 1, j) - Ez(i, j)),  i < nx - 1,
                 *
                 * with the coefficients of cell (i, j). Hx on the last row and Hy on the last
                 * column lie outside the domain and are left untouched. Inner loop along x.
                 *
                 * @tparam Simd Whether the inner loops carry `omp simd`.
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid.
                 */
                template <bool Simd = false>
                inline void update_h_block(Fields &fields, CoeffsView coeffs,
                                           const Block &block) noexcept
                {
                    const Grid &grid = fields.grid();
                    const std::size_t pitch = grid.pitch();
                    const std::size_t hxEnd = std::min(block.j1, grid.ny() - 1);
                    const std::size_t hyEnd = std::min(block.i1, grid.nx() - 1);
                    const FloatType invDx = FloatType(1) / grid.dx();
                    const FloatType invDy = FloatType(1) / grid.dy();

                    const FloatType *__restrict Ez = fields.ez().data();
                    FloatType *__restrict Hx = fields.hx().data();
                    FloatType *__restrict Hy = fields.hy().data();
                    const FloatType *__restrict Chx1 = coeffs.coeff_hx_1;
                    const FloatType *__restrict Chx2 = coeffs.coeff_hx_2;
                    const FloatType *__restrict Chy1 = coeffs.coeff_hy_1;
                    const FloatType *__restrict Chy2 = coeffs.coeff_hy_2;

                    for (std::size_t j = block.j0; j < hxEnd; ++j) {
                        const std::size_t row = j * pitch;
                        for_range<Simd>(block.i0, block.i1, [=](std::size_t i) {
                            const std::size_t k = row + i;
                            Hx[k] = Chx1[k] * Hx[k] - Chx2[k] * invDy * (Ez[k + pitch] - Ez[k]);
                        });
                    }

                    for (std::size_t j = block.j0; j < block.j1; ++j) {
                        const std::size_t row = j * pitch;
                        for_range<Simd>(block.i0, hyEnd, [=](std::size_t i) {
                            const std::size_t k = row + i;
                            Hy[k] = Chy1[k] * Hy[k] + Chy2[k] * invDx * (Ez[k + 1] - Ez[k]);
                        });
                    }
                }

                /**
                 * Block of the cells whose Ez takes the plain update: inside the PML and off the
                 * edge.
                 *
                 * @param grid Grid of the fields.
                 * @param pml PML thickness in cells, zero for none.
                 * @return max(pml, 1) <= i < nx - max(pml, 1), and the same along j.
                 */
                inline Block interior_block(const Grid &grid, std::size_t pml) noexcept
                {
                    const std::size_t lo = std::max<std::size_t>(pml, 1);
                    return {
                        .i0 = lo,
                        .i1 = grid.nx() - std::min(lo, grid.nx()),
                        .j0 = lo,
                        .j1 = grid.ny() - std::min(lo, grid.ny()),
                    };
                }

                /**
                 * Advance Ez over the part of a block inside the PML by one step from the curl of
                 * H.
                 *
                 *     Ez(i, j) = Cex1 Ez(i, j) + Cex2 ((Hy(i, j) - Hy(i - 1, j)) / dx
                 *                                    - (Hx(i, j) - Hx(i, j - 1)) / dy)
                 *
                 * over the cells of the block in interior_block(), with the x coefficients, which
                 * equal the y ones there. With no PML that is the whole interior. The PML is left
                 * to update_pml_block, the edge Ez to update_boundary_block. Inner loop along x.
                 *
                 * @tparam Simd Whether the inner loop carries `omp simd`.
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid.
                 * @param pml PML thickness in cells, zero for none.
                 */
                template <bool Simd = false>
                inline void update_e_interior_block(Fields &fields, CoeffsView coeffs,
                                                    const Block &block, std::size_t pml) noexcept
                {
                    const Grid &grid = fields.grid();
                    const Block cells = intersect(block, interior_block(grid, pml));
                    if (cells.empty()) {
                        return;
                    }
                    const std::size_t pitch = grid.pitch();
                    const FloatType invDx = FloatType(1) / grid.dx();
                    const FloatType invDy = FloatType(1) / grid.dy();

                    FloatType *__restrict Ez = fields.ez().data();
                    const FloatType *__restrict Hx = fields.hx().data();
                    const FloatType *__restrict Hy = fields.hy().data();
                    const FloatType *__restrict Cex1 = coeffs.coeff_ez_x1;
                    const FloatType *__restrict Cex2 = coeffs.coeff_ez_x2;

                    for (std::size_t j = cells.j0; j < cells.j1; ++j) {
                        const std::size_t row = j * pitch;
                        for_range<Simd>(cells.i0, cells.i1, [=](std::size_t i) {
                            const std::size_t k = row + i;
                            const FloatType curl =
                                (Hy[k] - Hy[k - 1]) * invDx - (Hx[k] - Hx[k - pitch]) * invDy;
                            Ez[k] = Cex1[k] * Ez[k] + Cex2[k] * curl;
                        });
                    }
                }

                /**
                 * Advance Ezx and Ezy over a rectangle of non-edge cells, and Ez as their sum.
                 *
                 * @tparam Simd Whether the inner loop carries `omp simd`.
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param cells Cells to update, off the edge.
                 */
                template <bool Simd = false>
                inline void update_split_rect(Fields &fields, CoeffsView coeffs,
                                              const Block &cells) noexcept
                {
                    if (cells.empty()) {
                        return;
                    }
                    const Grid &grid = fields.grid();
                    const std::size_t pitch = grid.pitch();
                    const FloatType invDx = FloatType(1) / grid.dx();
                    const FloatType invDy = FloatType(1) / grid.dy();

                    FloatType *__restrict Ez = fields.ez().data();
                    FloatType *__restrict Ezx = fields.ezx().data();
                    FloatType *__restrict Ezy = fields.ezy().data();
                    const FloatType *__restrict Hx = fields.hx().data();
                    const FloatType *__restrict Hy = fields.hy().data();
                    const FloatType *__restrict Cex1 = coeffs.coeff_ez_x1;
                    const FloatType *__restrict Cex2 = coeffs.coeff_ez_x2;
                    const FloatType *__restrict Cey1 = coeffs.coeff_ez_y1;
                    const FloatType *__restrict Cey2 = coeffs.coeff_ez_y2;

                    for (std::size_t j = cells.j0; j < cells.j1; ++j) {
                        const std::size_t row = j * pitch;
                        for_range<Simd>(cells.i0, cells.i1, [=](std::size_t i) {
                            const std::size_t k = row + i;
                            Ezx[k] = Cex1[k] * Ezx[k] + Cex2[k] * invDx * (Hy[k] - Hy[k - 1]);
                            Ezy[k] = Cey1[k] * Ezy[k] - Cey2[k] * invDy * (Hx[k] - Hx[k - pitch]);
                            Ez[k] = Ezx[k] + Ezy[k];
                        });
                    }
                }

                /**
                 * Advance Ez over the part of a block in the PML by one step as Ezx and Ezy.
                 *
                 *     Ezx(i, j) = Cex1 Ezx(i, j) + Cex2 / dx (Hy(i, j) - Hy(i - 1, j)),
                 *     Ezy(i, j) = Cey1 Ezy(i, j) - Cey2 / dy (Hx(i, j) - Hx(i, j - 1)),
                 *     Ez(i, j)  = Ezx(i, j) + Ezy(i, j),
                 *
                 * over the cells of the block that are neither in interior_block() nor on the edge:
                 * the bottom and top slabs of pml - 1 rows across the width, then the left and
                 * right slabs of pml - 1 columns between them. The corners lie in the bottom and
                 * top slabs and carry both conductivities through the coefficients. Ez there is
                 * rebuilt from its parts each step, so fields put in by hand keep Ez = Ezx + Ezy
                 * inside the PML. Inner loop along x.
                 *
                 * @tparam Simd Whether the inner loops carry `omp simd`.
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid.
                 * @param pml PML thickness in cells, with 2 pml below nx and ny; zero for none.
                 */
                template <bool Simd = false>
                inline void update_pml_block(Fields &fields, CoeffsView coeffs, const Block &block,
                                             std::size_t pml) noexcept
                {
                    if (pml < 2) {
                        return;
                    }
                    const std::size_t nx = fields.grid().nx();
                    const std::size_t ny = fields.grid().ny();
                    const Block bottom{.i0 = 1, .i1 = nx - 1, .j0 = 1, .j1 = pml};
                    const Block top{.i0 = 1, .i1 = nx - 1, .j0 = ny - pml, .j1 = ny - 1};
                    const Block left{.i0 = 1, .i1 = pml, .j0 = pml, .j1 = ny - pml};
                    const Block right{.i0 = nx - pml, .i1 = nx - 1, .j0 = pml, .j1 = ny - pml};
                    update_split_rect<Simd>(fields, coeffs, intersect(block, bottom));
                    update_split_rect<Simd>(fields, coeffs, intersect(block, top));
                    update_split_rect<Simd>(fields, coeffs, intersect(block, left));
                    update_split_rect<Simd>(fields, coeffs, intersect(block, right));
                }

                /**
                 * Update the edge Ez of a block, which update_e_interior_block and update_pml_block
                 * leave alone; call it right after them.
                 *
                 * Under PEC and PML the edge Ez is set to zero, with Ezx and Ezy. Under PMC it
                 * takes the interior update, with the x coefficients, and every H outside the
                 * domain read as zero: Hy(-1, j) and Hy(nx - 1, j), Hx(i, -1) and Hx(i, ny - 1).
                 * Corners combine both.
                 *
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid; only its edge cells change.
                 * @param boundary Outer boundary condition.
                 */
                void update_boundary_block(Fields &fields, CoeffsView coeffs, const Block &block,
                                           Boundary boundary) noexcept;

                /**
                 * Advance every Ez of a block by one step: the interior, the PML and the edge.
                 *
                 * @tparam Simd Whether the inner loops carry `omp simd`.
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid.
                 * @param pml PML thickness in cells, zero for none.
                 * @param boundary Outer boundary condition.
                 */
                template <bool Simd = false>
                inline void update_e_block(Fields &fields, CoeffsView coeffs, const Block &block,
                                           std::size_t pml, Boundary boundary) noexcept
                {
                    update_e_interior_block<Simd>(fields, coeffs, block, pml);
                    update_pml_block<Simd>(fields, coeffs, block, pml);
                    update_boundary_block(fields, coeffs, block, boundary);
                }
            }
        }
    }
}

#endif
