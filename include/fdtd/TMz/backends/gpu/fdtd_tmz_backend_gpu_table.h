/*
 * C++ header for the CUDA backend's table of distinct update coefficients, read in place of arrays.
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

#ifndef FDTD_TMZ_BACKEND_GPU_TABLE_H
#define FDTD_TMZ_BACKEND_GPU_TABLE_H

#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <cstdint>
#include <span>
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
            /** Namespace for what the 2D TMz CUDA backend's files share. */
            namespace gpu
            {
                /**
                 * The update coefficients of a cell whose Ez takes the same pair along x and y, as every cell outside the PML does: the
                 * pairs of Ez, Hx and Hy.
                 */
                struct CoeffTuple
                {
                    /** Ez decay factor, along x and y alike. */
                    FloatType ez_1;

                    /** Ez curl factor, along x and y alike, before 1/dx or 1/dy. */
                    FloatType ez_2;

                    /** Hx decay factor. */
                    FloatType hx_1;

                    /** Hx curl factor, before 1/dy. */
                    FloatType hx_2;

                    /** Hy decay factor. */
                    FloatType hy_1;

                    /** Hy curl factor, before 1/dx. */
                    FloatType hy_2;
                };

                /**
                 * The distinct coefficient tuples of a grid and the entry each cell takes, for kernels that read a few tuples instead of
                 * eight arrays per cell.
                 *
                 * A cell takes an entry when its Ez pairs along x and y are equal; the entries hold the cells' values exactly, compared bit
                 * by bit, so a kernel that reads them computes what it would from the arrays. A cell whose pairs differ, as in the PML, or
                 * whose tuple finds no room once the table holds CAPACITY entries, takes ARRAYS and reads the arrays.
                 */
                class CoeffTable
                {
                public:
                    /** Most entries a table holds, so an index fits a byte with ARRAYS to spare. */
                    static constexpr std::size_t CAPACITY = 255;

                    /** Index of a cell that reads the arrays. */
                    static constexpr std::uint8_t ARRAYS = 255;

                    /**
                     * Find the distinct tuples of the coefficients and the entry of every cell.
                     *
                     * Entries are numbered in the order their first cell comes in, row by row.
                     *
                     * @param coeffs Coefficients to read.
                     */
                    explicit CoeffTable(const Coeffs &coeffs);

                    /**
                     * Grid of the coefficients.
                     *
                     * @return The grid.
                     */
                    const Grid &grid() const noexcept;

                    /**
                     * The distinct tuples.
                     *
                     * @return At most CAPACITY entries.
                     */
                    std::span<const CoeffTuple> entries() const noexcept;

                    /**
                     * The entry of each cell, or ARRAYS, indexed with Grid::index.
                     *
                     * @return One index per element of the grid.
                     */
                    std::span<const std::uint8_t> index() const noexcept;

                    /**
                     * The one entry every cell of the core of an interior shares.
                     *
                     * The core is the interior less its last column and row, where Hy and Hx sit half a cell into the PML and take its
                     * grading.
                     *
                     * @param lo Lower bound of the interior along both axes: the core holds the
                     * cells with lo <= i < nx - lo - 1 and lo <= j < ny - lo - 1.
                     * @return The entry, or null when the core is empty or its cells take more than
                     * one entry or ARRAYS.
                     */
                    const CoeffTuple *uniform_entry(std::size_t lo) const noexcept;

                private:
                    /** Grid of the coefficients. */
                    Grid m_grid;

                    /** The distinct tuples. */
                    std::vector<CoeffTuple> m_entries;

                    /** The entry of each cell. */
                    std::vector<std::uint8_t> m_index;
                };
            }
        }
    }
}

#endif
