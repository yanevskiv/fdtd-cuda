/*
 * C++ header for the serial 2D TMz Yee update.
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

#ifndef FDTD_TMZ_SERIAL_UPDATE_H
#define FDTD_TMZ_SERIAL_UPDATE_H

#include "fdtd/TMz/fdtd_tmz_coeffs2d.h"
#include "fdtd/TMz/fdtd_tmz_fields2d.h"

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /**
         * Advance Hx and Hy by one step from the differences of Ez.
         *
         * With Hx(i, j) at (i, j + 1/2) and Hy(i, j) at (i + 1/2, j):
         *
         *     Hx(i, j) = Ch1 Hx(i, j) - Ch2 / dy (Ez(i, j + 1) - Ez(i, j)),  j < ny - 1,
         *     Hy(i, j) = Ch1 Hy(i, j) + Ch2 / dx (Ez(i + 1, j) - Ez(i, j)),  i < nx - 1,
         *
         * with the coefficients of cell (i, j). Hx on the last row and Hy on the last
         * column lie outside the domain and are left untouched. Inner loop along x.
         *
         * @param fields Fields to update.
         * @param coeffs Coefficients on the same grid as the fields.
         */
        void serial_update_h(Fields2D &fields, const Coeffs2D &coeffs) noexcept;

        /**
         * Advance Ez by one step from the curl of H.
         *
         *     Ez(i, j) = Ce1 Ez(i, j) + Ce2 ((Hy(i, j) - Hy(i - 1, j)) / dx
         *                                  - (Hx(i, j) - Hx(i, j - 1)) / dy)
         *
         * over the interior 0 < i < nx - 1, 0 < j < ny - 1. The edge Ez is left to the
         * outer boundary. Inner loop along x.
         *
         * @param fields Fields to update.
         * @param coeffs Coefficients on the same grid as the fields.
         */
        void serial_update_e(Fields2D &fields, const Coeffs2D &coeffs) noexcept;
    }
}

#endif
