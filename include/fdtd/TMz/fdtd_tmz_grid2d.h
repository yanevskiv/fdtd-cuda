/*
 * C++ header for the 2D TMz grid.
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

#ifndef FDTD_TMZ_GRID2D_H
#define FDTD_TMZ_GRID2D_H

#include "fdtd/fdtd_float.h"

#include <cstddef>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /**
         * Geometry of a 2D Yee grid: cell counts, cell sizes and row-major indexing.
         *
         * Cell (i, j) is at index j * pitch + i, with x fastest. The pitch equals nx
         * on the CPU; a GPU backend may pad it for coalescing.
         */
        class Grid2D
        {
        public:
            /**
             * Create a grid.
             *
             * @param nx Cells along x, at least 1.
             * @param ny Cells along y, at least 1.
             * @param dx Cell size along x in meters, positive.
             * @param dy Cell size along y in meters, positive.
             * @throws std::invalid_argument On a zero count, a non-positive size or an
             *         index range that does not fit std::size_t.
             */
            Grid2D(std::size_t nx, std::size_t ny, FloatType dx = 1, FloatType dy = 1);

            /**
             * Cells along x.
             *
             * @return Cell count.
             */
            std::size_t nx() const noexcept;

            /**
             * Cells along y.
             *
             * @return Cell count.
             */
            std::size_t ny() const noexcept;

            /**
             * Cell size along x.
             *
             * @return Size in meters.
             */
            FloatType dx() const noexcept;

            /**
             * Cell size along y.
             *
             * @return Size in meters.
             */
            FloatType dy() const noexcept;

            /**
             * Distance in elements between the starts of two consecutive rows.
             *
             * @return Row pitch.
             */
            std::size_t pitch() const noexcept;

            /**
             * Elements an array over this grid holds, padding included.
             *
             * @return pitch() * ny().
             */
            std::size_t size() const noexcept;

            /**
             * Index of cell (i, j). Unchecked and inline, as it sits on the update path.
             *
             * @param i Cell along x, below nx().
             * @param j Cell along y, below ny().
             * @return j * pitch() + i.
             */
            inline std::size_t index(std::size_t i, std::size_t j) const noexcept
            {
                return j * m_pitch + i;
            }

        private:
            /** Cells along x. */
            std::size_t m_nx;

            /** Cells along y. */
            std::size_t m_ny;

            /** Cell size along x in meters. */
            FloatType m_dx;

            /** Cell size along y in meters. */
            FloatType m_dy;

            /** Row pitch in elements. */
            std::size_t m_pitch;
        };

        /**
         * Time step for a Courant number on a grid in vacuum.
         *
         * The 2D stability limit is dt <= 1 / (c0 * sqrt(1/dx^2 + 1/dy^2)); the result is
         * that limit scaled by the Courant number. Computed in double.
         *
         * @param grid Grid whose cell sizes set the limit.
         * @param courant Courant number S, with 0 < S <= 1.
         * @return Time step in seconds.
         * @throws std::invalid_argument When S is outside (0, 1].
         */
        double courant_dt(const Grid2D &grid, double courant);
    }
}

#endif
