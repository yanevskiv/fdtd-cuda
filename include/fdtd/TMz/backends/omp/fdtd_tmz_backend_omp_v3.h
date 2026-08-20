/*
 * C++ header for the 2D TMz OpenMP backend's level v3, table.
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

#ifndef FDTD_TMZ_BACKEND_OMP_V3_H
#define FDTD_TMZ_BACKEND_OMP_V3_H

#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_levels.h"
#include "fdtd/TMz/backends/omp/fdtd_tmz_backend_omp_v2.h"
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
                 * Namespace for level v3: the v2 kernels with the core cells reading the tuple
                 * they share instead of the arrays.
                 *
                 * The tuple holds the bits the arrays hold, so the results are v2's bit for bit.
                 * The PML and the edge read the arrays as in v2.
                 */
                namespace v3
                {
                    /**
                     * Call a body for every index of a range as one `omp simd` loop.
                     *
                     * @tparam Body Callable taking the index.
                     * @param begin First index.
                     * @param end One past the last index.
                     * @param body Body of the loop, free of dependences between indices.
                     */
                    template <typename Body>
                    inline void for_simd(std::size_t begin, std::size_t end, Body body) noexcept
                    {
                        #pragma omp simd
                        for (std::size_t i = begin; i < end; ++i) {
                            body(i);
                        }
                    }

                    /**
                     * Call one of two bodies for every index of a row: the tuple body on the
                     * cells of the core, the arrays body on the others.
                     *
                     * @tparam Arrays Callable taking the index, reading the arrays.
                     * @tparam Tuple Callable taking the index, reading the shared tuple.
                     * @param shared The tuple and its core, or null for none.
                     * @param j Row of the indices.
                     * @param begin First index.
                     * @param end One past the last index.
                     * @param arrays Body for the cells that read the arrays.
                     * @param tuple Body for the cells that read the tuple.
                     */
                    template <typename Arrays, typename Tuple>
                    inline void for_row(const SharedTuple *shared, std::size_t j, std::size_t begin,
                                        std::size_t end, Arrays arrays, Tuple tuple) noexcept
                    {
                        if (shared != nullptr && j >= shared->core.j0 && j < shared->core.j1) {
                            const std::size_t first =
                                std::min(std::max(shared->core.i0, begin), end);
                            const std::size_t last =
                                std::min(std::max(shared->core.i1, first), end);
                            for_simd(begin, first, arrays);
                            for_simd(first, last, tuple);
                            for_simd(last, end, arrays);
                            return;
                        }
                        for_simd(begin, end, arrays);
                    }

                    /**
                     * The shared tuple, or zeros for none.
                     *
                     * @param shared The tuple and its core, or null for none.
                     * @return A copy of the tuple.
                     */
                    inline CoeffTuple tuple_or_zero(const SharedTuple *shared) noexcept
                    {
                        return shared != nullptr ? shared->tuple : CoeffTuple{};
                    }

                    /**
                     * Advance Hx and Hy over a block, as v2::update_h_block(), the core reading
                     * the tuple.
                     *
                     * @param fields Fields to update.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param shared The tuple and its core, or null for none.
                     * @param block Cells to update, inside the grid.
                     */
                    inline void update_h_block(Fields &fields, serial::CoeffsView coeffs,
                                               const SharedTuple *shared,
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
                        const CoeffTuple tuple = tuple_or_zero(shared);

                        for (std::size_t j = block.j0; j < hxEnd; ++j) {
                            const std::size_t row = j * pitch;
                            for_row(
                                shared, j, block.i0, block.i1,
                                [=](std::size_t i) {
                                    const std::size_t k = row + i;
                                    Hx[k] =
                                        Chx1[k] * Hx[k] - Chx2[k] * invDy * (Ez[k + pitch] - Ez[k]);
                                },
                                [=](std::size_t i) {
                                    const std::size_t k = row + i;
                                    Hx[k] = tuple.hx_1 * Hx[k] -
                                            tuple.hx_2 * invDy * (Ez[k + pitch] - Ez[k]);
                                });
                        }

                        for (std::size_t j = block.j0; j < block.j1; ++j) {
                            const std::size_t row = j * pitch;
                            for_row(
                                shared, j, block.i0, hyEnd,
                                [=](std::size_t i) {
                                    const std::size_t k = row + i;
                                    Hy[k] = Chy1[k] * Hy[k] + Chy2[k] * invDx * (Ez[k + 1] - Ez[k]);
                                },
                                [=](std::size_t i) {
                                    const std::size_t k = row + i;
                                    Hy[k] = tuple.hy_1 * Hy[k] +
                                            tuple.hy_2 * invDx * (Ez[k + 1] - Ez[k]);
                                });
                        }
                    }

                    /**
                     * Advance Ez over the part of a block inside the PML, as
                     * v2::update_e_interior_block(), the core reading the tuple.
                     *
                     * @param fields Fields to update.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param shared The tuple and its core, or null for none.
                     * @param block Cells to update, inside the grid.
                     * @param pml PML thickness in cells, zero for none.
                     */
                    inline void update_e_interior_block(Fields &fields, serial::CoeffsView coeffs,
                                                        const SharedTuple *shared,
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
                        const CoeffTuple tuple = tuple_or_zero(shared);

                        for (std::size_t j = cells.j0; j < cells.j1; ++j) {
                            const std::size_t row = j * pitch;
                            for_row(
                                shared, j, cells.i0, cells.i1,
                                [=](std::size_t i) {
                                    const std::size_t k = row + i;
                                    const FloatType curl = (Hy[k] - Hy[k - 1]) * invDx -
                                                           (Hx[k] - Hx[k - pitch]) * invDy;
                                    Ez[k] = Cex1[k] * Ez[k] + Cex2[k] * curl;
                                },
                                [=](std::size_t i) {
                                    const std::size_t k = row + i;
                                    const FloatType curl = (Hy[k] - Hy[k - 1]) * invDx -
                                                           (Hx[k] - Hx[k - pitch]) * invDy;
                                    Ez[k] = tuple.ez_1 * Ez[k] + tuple.ez_2 * curl;
                                });
                        }
                    }

                    /**
                     * Advance every Ez of a block: the interior with the tuple, the PML as v2,
                     * then the edge with the serial kernel.
                     *
                     * @param fields Fields to update.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param shared The tuple and its core, or null for none.
                     * @param block Cells to update, inside the grid.
                     * @param pml PML thickness in cells, zero for none.
                     * @param boundary Outer boundary condition.
                     */
                    inline void update_e_block(Fields &fields, serial::CoeffsView coeffs,
                                               const SharedTuple *shared,
                                               const serial::Block &block, std::size_t pml,
                                               Boundary boundary) noexcept
                    {
                        v3::update_e_interior_block(fields, coeffs, shared, block, pml);
                        v2::update_pml_block(fields, coeffs, block, pml);
                        serial::update_boundary_block(fields, coeffs, block, boundary);
                    }

                    /**
                     * Run the H phase of a part in step n with these kernels.
                     *
                     * @param stepper The advance.
                     * @param shared The tuple and its core, or null for none.
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    inline void h_phase(const serial::Stepper &stepper, const SharedTuple *shared,
                                        std::size_t part, std::size_t n) noexcept
                    {
                        for (const serial::Block &block : stepper.partition().blocks(part)) {
                            v3::update_h_block(stepper.fields(), stepper.coeffs(), shared, block);
                        }
                        stepper.add_sources(part, n, serial::SourceField::Magnetic);
                    }

                    /**
                     * Run the E phase of a part in step n with these kernels.
                     *
                     * @param stepper The advance.
                     * @param shared The tuple and its core, or null for none.
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    inline void e_phase(const serial::Stepper &stepper, const SharedTuple *shared,
                                        std::size_t part, std::size_t n) noexcept
                    {
                        for (const serial::Block &block : stepper.partition().blocks(part)) {
                            v3::update_e_block(stepper.fields(), stepper.coeffs(), shared, block,
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
