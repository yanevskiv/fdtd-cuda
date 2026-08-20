/*
 * C++ header for the 2D TMz OpenMP backend's level v2, simd.
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

#ifndef FDTD_TMZ_BACKEND_OMP_V2_H
#define FDTD_TMZ_BACKEND_OMP_V2_H

#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"
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
            /** Namespace for the levels of the 2D TMz OpenMP backend. */
            namespace omp
            {
                /**
                 * Namespace for level v2: the serial kernels with `omp simd` on their inner loops.
                 *
                 * The arithmetic is the serial kernels', so the results are theirs bit for bit
                 * in the exact build.
                 */
                namespace v2
                {
                    /**
                     * Advance Hx and Hy over a block, as serial::update_h_block().
                     *
                     * @param fields Fields to update.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param block Cells to update, inside the grid.
                     */
                    inline void update_h_block(Fields &fields, serial::CoeffsView coeffs,
                                               const serial::Block &block) noexcept
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

                            #pragma omp simd
                            for (std::size_t i = block.i0; i < block.i1; ++i) {
                                const std::size_t k = row + i;
                                Hx[k] = Chx1[k] * Hx[k] - Chx2[k] * invDy * (Ez[k + pitch] - Ez[k]);
                            }
                        }

                        for (std::size_t j = block.j0; j < block.j1; ++j) {
                            const std::size_t row = j * pitch;

                            #pragma omp simd
                            for (std::size_t i = block.i0; i < hyEnd; ++i) {
                                const std::size_t k = row + i;
                                Hy[k] = Chy1[k] * Hy[k] + Chy2[k] * invDx * (Ez[k + 1] - Ez[k]);
                            }
                        }
                    }

                    /**
                     * Advance Ez over the part of a block inside the PML, as
                     * serial::update_e_interior_block().
                     *
                     * @param fields Fields to update.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param block Cells to update, inside the grid.
                     * @param pml PML thickness in cells, zero for none.
                     */
                    inline void update_e_interior_block(Fields &fields, serial::CoeffsView coeffs,
                                                        const serial::Block &block,
                                                        std::size_t pml) noexcept
                    {
                        const Grid &grid = fields.grid();
                        const serial::Block cells =
                            serial::intersect(block, serial::interior_block(grid, pml));
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

                            #pragma omp simd
                            for (std::size_t i = cells.i0; i < cells.i1; ++i) {
                                const std::size_t k = row + i;
                                const FloatType curl =
                                    (Hy[k] - Hy[k - 1]) * invDx - (Hx[k] - Hx[k - pitch]) * invDy;
                                Ez[k] = Cex1[k] * Ez[k] + Cex2[k] * curl;
                            }
                        }
                    }

                    /**
                     * Advance Ezx, Ezy and Ez over a rectangle of non-edge cells, as
                     * serial::update_split_rect().
                     *
                     * @param fields Fields to update.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param cells Cells to update, off the edge.
                     */
                    inline void update_split_rect(Fields &fields, serial::CoeffsView coeffs,
                                                  const serial::Block &cells) noexcept
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

                            #pragma omp simd
                            for (std::size_t i = cells.i0; i < cells.i1; ++i) {
                                const std::size_t k = row + i;
                                Ezx[k] = Cex1[k] * Ezx[k] + Cex2[k] * invDx * (Hy[k] - Hy[k - 1]);
                                Ezy[k] =
                                    Cey1[k] * Ezy[k] - Cey2[k] * invDy * (Hx[k] - Hx[k - pitch]);
                                Ez[k] = Ezx[k] + Ezy[k];
                            }
                        }
                    }

                    /**
                     * Advance Ez over the part of a block in the PML, as
                     * serial::update_pml_block(): the bottom and top slabs, then the left and
                     * right ones between them.
                     *
                     * @param fields Fields to update.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param block Cells to update, inside the grid.
                     * @param pml PML thickness in cells, with 2 pml below nx and ny; zero for none.
                     */
                    inline void update_pml_block(Fields &fields, serial::CoeffsView coeffs,
                                                 const serial::Block &block,
                                                 std::size_t pml) noexcept
                    {
                        if (pml < 2) {
                            return;
                        }
                        const std::size_t nx = fields.grid().nx();
                        const std::size_t ny = fields.grid().ny();
                        const serial::Block bottom{.i0 = 1, .i1 = nx - 1, .j0 = 1, .j1 = pml};
                        const serial::Block top{
                            .i0 = 1, .i1 = nx - 1, .j0 = ny - pml, .j1 = ny - 1};
                        const serial::Block left{.i0 = 1, .i1 = pml, .j0 = pml, .j1 = ny - pml};
                        const serial::Block right{
                            .i0 = nx - pml, .i1 = nx - 1, .j0 = pml, .j1 = ny - pml};
                        v2::update_split_rect(fields, coeffs, serial::intersect(block, bottom));
                        v2::update_split_rect(fields, coeffs, serial::intersect(block, top));
                        v2::update_split_rect(fields, coeffs, serial::intersect(block, left));
                        v2::update_split_rect(fields, coeffs, serial::intersect(block, right));
                    }

                    /**
                     * Advance every Ez of a block: the interior and the PML with SIMD, then the
                     * edge with the serial kernel.
                     *
                     * @param fields Fields to update.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param block Cells to update, inside the grid.
                     * @param pml PML thickness in cells, zero for none.
                     * @param boundary Outer boundary condition.
                     */
                    inline void update_e_block(Fields &fields, serial::CoeffsView coeffs,
                                               const serial::Block &block, std::size_t pml,
                                               Boundary boundary) noexcept
                    {
                        v2::update_e_interior_block(fields, coeffs, block, pml);
                        v2::update_pml_block(fields, coeffs, block, pml);
                        serial::update_boundary_block(fields, coeffs, block, boundary);
                    }

                    /**
                     * Run the H phase of a part in step n with these kernels, as
                     * serial::Stepper::h_phase().
                     *
                     * @param stepper The advance.
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    inline void h_phase(const serial::Stepper &stepper, std::size_t part,
                                        std::size_t n) noexcept
                    {
                        for (const serial::Block &block : stepper.partition().blocks(part)) {
                            v2::update_h_block(stepper.fields(), stepper.coeffs(), block);
                        }
                        stepper.add_sources(part, n, serial::SourceField::Magnetic);
                    }

                    /**
                     * Run the E phase of a part in step n with these kernels, as
                     * serial::Stepper::e_phase().
                     *
                     * @param stepper The advance.
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    inline void e_phase(const serial::Stepper &stepper, std::size_t part,
                                        std::size_t n) noexcept
                    {
                        for (const serial::Block &block : stepper.partition().blocks(part)) {
                            v2::update_e_block(stepper.fields(), stepper.coeffs(), block,
                                               stepper.pml(), stepper.boundary());
                        }
                        stepper.add_sources(part, n, serial::SourceField::Electric);
                        stepper.record_probes(part, n);
                    }
                }
            }
        }
    }
}

#endif
