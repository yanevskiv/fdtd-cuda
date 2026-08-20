/*
 * C++ header for the 2D TMz serial backend: the stencil kernels, the partition and the stepper.
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

#ifndef FDTD_TMZ_BACKEND_SERIAL_H
#define FDTD_TMZ_BACKEND_SERIAL_H

#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/fdtd_aligned_buffer.h"
#include "fdtd/fdtd_float.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /** Namespace for the serial FDTD algorithm, which the OpenMP backend runs in parts. */
            namespace serial
            {
                /** Fields a set of sources adds to. */
                enum class SourceField {
                    /** Ez. */
                    Electric,

                    /** Hx and Hy. */
                    Magnetic
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
                 * Pointers to the eight coefficient arrays the update kernels read.
                 *
                 * The kernels take these rather than a Coeffs so a backend can keep its own arrays. A Coeffs converts implicitly.
                 */
                struct CoeffsView
                {
                    /**
                     * View the arrays of a coefficient set, which must outlive the view.
                     *
                     * @param coeffs Coefficients to view.
                     */
                    CoeffsView(const Coeffs &coeffs) noexcept; // NOLINT(google-explicit-constructor)

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
                    CoeffsView(const FloatType *ezX1, const FloatType *ezX2, const FloatType *ezY1, const FloatType *ezY2,
                               const FloatType *hx1, const FloatType *hx2, const FloatType *hy1, const FloatType *hy2) noexcept;

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
                 * Advance Hx and Hy over a block by one step from the differences of Ez.
                 *
                 * With Hx(i, j) at (i, j + 1/2) and Hy(i, j) at (i + 1/2, j):
                 *
                 *     Hx(i, j) = Chx1 Hx(i, j) - Chx2 / dy (Ez(i, j + 1) - Ez(i, j)),  j < ny - 1,
                 *     Hy(i, j) = Chy1 Hy(i, j) + Chy2 / dx (Ez(i + 1, j) - Ez(i, j)),  i < nx - 1,
                 *
                 * with the coefficients of cell (i, j). Hx on the last row and Hy on the last column lie outside the domain and are left
                 * untouched. Inner loop along x.
                 *
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid.
                 */
                inline void update_h_block(Fields &fields, CoeffsView coeffs, const Block &block) noexcept
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
                        for (std::size_t i = block.i0; i < block.i1; ++i) {
                            const std::size_t k = row + i;
                            Hx[k] = Chx1[k] * Hx[k] - Chx2[k] * invDy * (Ez[k + pitch] - Ez[k]);
                        }
                    }

                    for (std::size_t j = block.j0; j < block.j1; ++j) {
                        const std::size_t row = j * pitch;
                        for (std::size_t i = block.i0; i < hyEnd; ++i) {
                            const std::size_t k = row + i;
                            Hy[k] = Chy1[k] * Hy[k] + Chy2[k] * invDx * (Ez[k + 1] - Ez[k]);
                        }
                    }
                }

                /**
                 * Block of the cells whose Ez takes the plain update: inside the PML and off the edge.
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
                 * Advance Ez over the part of a block inside the PML by one step from the curl of H.
                 *
                 *     Ez(i, j) = Cex1 Ez(i, j) + Cex2 ((Hy(i, j) - Hy(i - 1, j)) / dx
                 *                                    - (Hx(i, j) - Hx(i, j - 1)) / dy)
                 *
                 * over the cells of the block in interior_block(), with the x coefficients, which equal the y ones there. With no PML that
                 * is the whole interior. The PML is left to update_pml_block, the edge Ez to update_boundary_block. Inner loop along x.
                 *
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid.
                 * @param pml PML thickness in cells, zero for none.
                 */
                inline void update_e_interior_block(Fields &fields, CoeffsView coeffs, const Block &block, std::size_t pml) noexcept
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
                        for (std::size_t i = cells.i0; i < cells.i1; ++i) {
                            const std::size_t k = row + i;
                            const FloatType curl = (Hy[k] - Hy[k - 1]) * invDx - (Hx[k] - Hx[k - pitch]) * invDy;
                            Ez[k] = Cex1[k] * Ez[k] + Cex2[k] * curl;
                        }
                    }
                }

                /**
                 * Advance Ezx and Ezy over a rectangle of non-edge cells, and Ez as their sum.
                 *
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param cells Cells to update, off the edge.
                 */
                inline void update_split_rect(Fields &fields, CoeffsView coeffs, const Block &cells) noexcept
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
                        for (std::size_t i = cells.i0; i < cells.i1; ++i) {
                            const std::size_t k = row + i;
                            Ezx[k] = Cex1[k] * Ezx[k] + Cex2[k] * invDx * (Hy[k] - Hy[k - 1]);
                            Ezy[k] = Cey1[k] * Ezy[k] - Cey2[k] * invDy * (Hx[k] - Hx[k - pitch]);
                            Ez[k] = Ezx[k] + Ezy[k];
                        }
                    }
                }

                /**
                 * Advance Ez over the part of a block in the PML by one step as Ezx and Ezy.
                 *
                 *     Ezx(i, j) = Cex1 Ezx(i, j) + Cex2 / dx (Hy(i, j) - Hy(i - 1, j)),
                 *     Ezy(i, j) = Cey1 Ezy(i, j) - Cey2 / dy (Hx(i, j) - Hx(i, j - 1)),
                 *     Ez(i, j)  = Ezx(i, j) + Ezy(i, j),
                 *
                 * over the cells of the block that are neither in interior_block() nor on the edge: the bottom and top slabs of pml - 1
                 * rows across the width, then the left and right slabs of pml - 1 columns between them. The corners lie in the bottom and
                 * top slabs and carry both conductivities through the coefficients. Ez there is rebuilt from its parts each step, so fields
                 * put in by hand keep Ez = Ezx + Ezy inside the PML. Inner loop along x.
                 *
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid.
                 * @param pml PML thickness in cells, with 2 pml below nx and ny; zero for none.
                 */
                inline void update_pml_block(Fields &fields, CoeffsView coeffs, const Block &block, std::size_t pml) noexcept
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
                    update_split_rect(fields, coeffs, intersect(block, bottom));
                    update_split_rect(fields, coeffs, intersect(block, top));
                    update_split_rect(fields, coeffs, intersect(block, left));
                    update_split_rect(fields, coeffs, intersect(block, right));
                }

                /**
                 * Update the edge Ez of a block, which update_e_interior_block and update_pml_block leave alone; call it right after them.
                 *
                 * Under PEC and PML the edge Ez is set to zero, with Ezx and Ezy. Under PMC it takes the interior update, with the x
                 * coefficients, and every H outside the domain read as zero: Hy(-1, j) and Hy(nx - 1, j), Hx(i, -1) and Hx(i, ny - 1).
                 * Corners combine both.
                 *
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid; only its edge cells change.
                 * @param boundary Outer boundary condition.
                 */
                void update_boundary_block(Fields &fields, CoeffsView coeffs, const Block &block, Boundary boundary) noexcept;

                /**
                 * Advance every Ez of a block by one step: the interior, the PML and the edge.
                 *
                 * @param fields Fields to update.
                 * @param coeffs Coefficient arrays on the same grid as the fields.
                 * @param block Cells to update, inside the grid.
                 * @param pml PML thickness in cells, zero for none.
                 * @param boundary Outer boundary condition.
                 */
                inline void update_e_block(Fields &fields, CoeffsView coeffs, const Block &block, std::size_t pml,
                                           Boundary boundary) noexcept
                {
                    update_e_interior_block(fields, coeffs, block, pml);
                    update_pml_block(fields, coeffs, block, pml);
                    update_boundary_block(fields, coeffs, block, boundary);
                }

                /**
                 * A cut of a grid into parts, each part one to three blocks, every cell in one part.
                 *
                 * A part is the unit of work a backend hands to a thread or a task. Tiles are a regular cut, rows being tiles of nx by 1,
                 * and know their neighbours.
                 */
                class Partition
                {
                public:
                    /**
                     * Cut a grid into one part, the whole grid.
                     *
                     * @param grid Grid to cut.
                     * @return The partition.
                     */
                    static Partition whole(const Grid &grid);

                    /**
                     * Cut a grid into tiles of tileX by tileY cells, row-major, the last ones clipped.
                     *
                     * @param grid Grid to cut.
                     * @param tileX Tile width in cells, positive.
                     * @param tileY Tile height in cells, positive.
                     * @return The partition.
                     * @throws std::invalid_argument When a tile side is zero.
                     */
                    static Partition tiles(const Grid &grid, std::size_t tileX, std::size_t tileY);

                    /**
                     * Number of parts.
                     *
                     * @return The part count.
                     */
                    std::size_t size() const noexcept;

                    /**
                     * Blocks of one part.
                     *
                     * @param part Part below size().
                     * @return The part's blocks, none for an empty range.
                     */
                    std::span<const Block> blocks(std::size_t part) const noexcept;

                    /**
                     * Part that holds a cell.
                     *
                     * @param i Column, below nx.
                     * @param j Row, below ny.
                     * @return The part.
                     */
                    std::size_t owner(std::size_t i, std::size_t j) const noexcept;

                    /**
                     * Tiles along x, part p sitting at column p % tiles_x() of the tiles.
                     *
                     * @return The tile column count.
                     */
                    std::size_t tiles_x() const noexcept;

                    /**
                     * Tiles along y.
                     *
                     * @return The tile row count.
                     */
                    std::size_t tiles_y() const noexcept;

                private:
                    /** Create a partition with no parts. */
                    Partition();

                    /**
                     * Close the part made of the blocks added since the last one.
                     */
                    void close_part();

                    /** Tile width. */
                    std::size_t m_tile_x = 0;

                    /** Tile height. */
                    std::size_t m_tile_y = 0;

                    /** Tile columns. */
                    std::size_t m_tiles_x = 0;

                    /** Tile rows. */
                    std::size_t m_tiles_y = 0;

                    /** Blocks of every part, part by part. */
                    std::vector<Block> m_blocks;

                    /** Part p's blocks run from m_offsets[p] to m_offsets[p + 1]. */
                    std::vector<std::size_t> m_offsets;
                };

                /**
                 * One advance of the fields of the serial or OpenMP backend, as two phases per part per step.
                 *
                 * Step n of a part is h_phase(part, n) then e_phase(part, n). The H phase updates the part's H and adds the H sources of
                 * its cells; the E phase updates its Ez, adds the Ez sources of its cells and writes the records of the probes on them.
                 * Whoever runs a phase owns the part's cells for that phase, so sources and probes need no lock: a backend only has to
                 * finish the H phase of a part's neighbours before its E phase, and the E phase of the neighbours before the next H phase.
                 * The serial backend runs the one part of Partition::whole. A backend with kernels of its own runs them over the part's
                 * blocks and calls add_sources() and record_probes() as the phases do.
                 */
                class Stepper
                {
                public:
                    /**
                     * Prepare an advance and sort the sources and probes by the part of their cell.
                     *
                     * @param fields Fields to advance; they must outlive the stepper.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param pml PML thickness in cells, zero for none.
                     * @param boundary Outer boundary condition.
                     * @param partition Parts of the grid; it must outlive the stepper.
                     * @param sources Distinct cells inside the grid to add to.
                     * @param values One value per source per step, step-major.
                     * @param probes Cells inside the grid to record.
                     * @param records Space for one value per probe per step, step-major: step n of probe k goes to records[n *
                     *        probes.size() + k].
                     */
                    Stepper(Fields &fields, CoeffsView coeffs, std::size_t pml, Boundary boundary, const Partition &partition,
                            std::span<const PointInjection> sources, std::span<const FloatType> values, std::span<const PointProbe> probes,
                            FloatType *records);

                    /**
                     * Run the H phase of a part in step n: its H, then the H sources on its cells.
                     *
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    inline void h_phase(std::size_t part, std::size_t n) const noexcept
                    {
                        for (const Block &block : m_partition.blocks(part)) {
                            update_h_block(m_fields, m_coeffs, block);
                        }
                        add_sources(part, n, SourceField::Magnetic);
                    }

                    /**
                     * Run the E phase of a part in step n: its Ez, the Ez sources on its cells, then the probes on its cells.
                     *
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    inline void e_phase(std::size_t part, std::size_t n) const noexcept
                    {
                        for (const Block &block : m_partition.blocks(part)) {
                            update_e_block(m_fields, m_coeffs, block, m_pml, m_boundary);
                        }
                        add_sources(part, n, SourceField::Electric);
                        record_probes(part, n);
                    }

                    /**
                     * Add step n's values to the sources of a part, the H ones or the Ez ones.
                     *
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     * @param field Fields the sources add to.
                     */
                    void add_sources(std::size_t part, std::size_t n, SourceField field) const noexcept;

                    /**
                     * Write step n's records of the probes of a part.
                     *
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    void record_probes(std::size_t part, std::size_t n) const noexcept;

                    /**
                     * Fields being advanced.
                     *
                     * @return The fields.
                     */
                    Fields &fields() const noexcept;

                    /**
                     * Coefficient arrays.
                     *
                     * @return The view.
                     */
                    CoeffsView coeffs() const noexcept;

                    /**
                     * PML thickness in cells.
                     *
                     * @return The thickness, zero for none.
                     */
                    std::size_t pml() const noexcept;

                    /**
                     * Outer boundary condition.
                     *
                     * @return The boundary.
                     */
                    Boundary boundary() const noexcept;

                    /**
                     * Parts of the grid.
                     *
                     * @return The partition.
                     */
                    const Partition &partition() const noexcept;

                private:
                    /** Fields being advanced. */
                    Fields &m_fields;

                    /** Coefficient arrays. */
                    CoeffsView m_coeffs;

                    /** PML thickness in cells, zero for none. */
                    std::size_t m_pml;

                    /** Outer boundary condition. */
                    Boundary m_boundary;

                    /** Parts of the grid. */
                    const Partition &m_partition;

                    /** Source cells. */
                    std::span<const PointInjection> m_sources;

                    /** Source values, step-major. */
                    std::span<const FloatType> m_values;

                    /** Probe cells. */
                    std::span<const PointProbe> m_probes;

                    /** Probe records of this advance, step-major. */
                    FloatType *m_records;

                    /** Sources of part p: m_source_order from m_source_offsets[p] to [p + 1]. */
                    std::vector<std::size_t> m_source_offsets;

                    /** Source indices sorted by part. */
                    std::vector<std::size_t> m_source_order;

                    /** Probes of part p: m_probe_order from m_probe_offsets[p] to [p + 1]. */
                    std::vector<std::size_t> m_probe_offsets;

                    /** Probe indices sorted by part. */
                    std::vector<std::size_t> m_probe_order;
                };
            }

            /**
             * Single-threaded reference backend, the oracle for the others: fields, coefficients and probe records in host memory, advanced
             * by a serial::Stepper over the one part of serial::Partition::whole.
             *
             * The OpenMP backend derives from it, keeps its storage and runs the same Stepper over more parts.
             */
            class SerialBackend : public Backend
            {
            public:
                /**
                 * Create the backend with zeroed fields.
                 *
                 * @param grid Grid the backend works on.
                 */
                explicit SerialBackend(const Grid &grid);

                /**
                 * Grid the backend was made for.
                 *
                 * @return The grid.
                 */
                const Grid &grid() const noexcept override;

                /**
                 * Copy fields into the backend.
                 *
                 * @param fields Fields on the backend's grid.
                 */
                void upload(const Fields &fields) override;

                /**
                 * Copy the backend's fields out.
                 *
                 * @param fields Destination on the backend's grid.
                 */
                void download(Fields &fields) const override;

                /**
                 * Copy update coefficients into the backend.
                 *
                 * @param coeffs Coefficients on the backend's grid.
                 */
                void upload(const Coeffs &coeffs) override;

                /**
                 * Set the outer boundary condition.
                 *
                 * @param boundary Outer boundary condition.
                 */
                void set_boundary(Boundary boundary) override;

                /**
                 * Outer boundary condition.
                 *
                 * @return The boundary.
                 */
                Boundary boundary() const noexcept override;

                /**
                 * Number of probes set.
                 *
                 * @return The probe count.
                 */
                std::size_t probe_count() const noexcept override;

                /**
                 * Number of steps recorded since the probes were set.
                 *
                 * @return The recorded step count.
                 */
                std::size_t probe_steps() const noexcept override;

                /**
                 * Copy the probe records out, step-major.
                 *
                 * @return The records.
                 */
                std::vector<FloatType> fetch_probes() const override;

                /**
                 * Modelled traffic of one step, two passes over per-cell coefficient arrays.
                 *
                 * @return two_pass_traffic() of the grid, the active PML and the boundary.
                 */
                StepTraffic step_traffic() const noexcept override;

                /**
                 * Short lowercase name of the backend.
                 *
                 * @return "serial".
                 */
                std::string_view name() const noexcept override;

                /**
                 * Type of the backend.
                 *
                 * @return BackendType::Serial.
                 */
                BackendType type() const noexcept override;

            protected:
                /**
                 * Advance the fields and add the sources, the whole grid as one part.
                 *
                 * @param nsteps Number of time steps.
                 * @param points Distinct cells inside the grid.
                 * @param values One value per point per step, step-major.
                 */
                void advance(std::size_t nsteps, std::span<const PointInjection> points, std::span<const FloatType> values) override;

                /**
                 * Replace the probes and clear the records.
                 *
                 * @param points Cells inside the grid.
                 */
                void reset_probes(std::span<const PointProbe> points) override;

                /**
                 * Check that coefficients were uploaded, before a step.
                 *
                 * @throws std::logic_error When none were.
                 */
                void require_coeffs() const;

                /**
                 * The backend's fields.
                 *
                 * @return The fields.
                 */
                Fields &fields() noexcept;

                /**
                 * The backend's coefficient arrays.
                 *
                 * @return A view of them.
                 */
                serial::CoeffsView coeffs() const noexcept;

                /**
                 * PML thickness the steps use: that of the coefficients under Boundary::Pml.
                 *
                 * @return The thickness, zero for none.
                 */
                std::size_t active_pml() const noexcept;

                /**
                 * Probe cells.
                 *
                 * @return The probes.
                 */
                std::span<const PointProbe> probes() const noexcept;

                /**
                 * Make room for the records of nsteps more steps.
                 *
                 * @param nsteps Steps about to run.
                 * @return Where step 0 of them goes, step-major as serial::Stepper writes them.
                 */
                FloatType *add_records(std::size_t nsteps);

            private:
                /** The backend's copy of the fields. */
                Fields m_fields;

                /** Whether coefficients were uploaded. */
                bool m_has_coeffs = false;

                /** PML thickness the uploaded coefficients carry, used under Boundary::Pml. */
                std::size_t m_pml_thickness = 0;

                /** The backend's copy of the Ez decay factors along x. */
                AlignedBuffer<FloatType> m_coeff_ez_x1;

                /** The backend's copy of the Ez curl factors along x. */
                AlignedBuffer<FloatType> m_coeff_ez_x2;

                /** The backend's copy of the Ez decay factors along y. */
                AlignedBuffer<FloatType> m_coeff_ez_y1;

                /** The backend's copy of the Ez curl factors along y. */
                AlignedBuffer<FloatType> m_coeff_ez_y2;

                /** The backend's copy of the Hx decay factors. */
                AlignedBuffer<FloatType> m_coeff_hx_1;

                /** The backend's copy of the Hx curl factors. */
                AlignedBuffer<FloatType> m_coeff_hx_2;

                /** The backend's copy of the Hy decay factors. */
                AlignedBuffer<FloatType> m_coeff_hy_1;

                /** The backend's copy of the Hy curl factors. */
                AlignedBuffer<FloatType> m_coeff_hy_2;

                /** Outer boundary condition. */
                Boundary m_boundary = Boundary::Pec;

                /** Probe cells. */
                std::vector<PointProbe> m_probes;

                /** Probe records, step-major. */
                std::vector<FloatType> m_records;
            };
        }
    }
}

#endif
