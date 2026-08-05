/*
 * C++ header for the 2D TMz Courant time step.
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

#ifndef FDTD_TMZ_COURANT_H
#define FDTD_TMZ_COURANT_H

#include "fdtd/TMz/fdtd_tmz_grid2d.h"

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
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
