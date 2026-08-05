/*
 * C++ header for the serial 2D TMz outer boundary.
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

#ifndef FDTD_TMZ_SERIAL_BOUNDARY_H
#define FDTD_TMZ_SERIAL_BOUNDARY_H

#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_coeffs2d.h"
#include "fdtd/TMz/fdtd_tmz_fields2d.h"

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /**
         * Update the edge Ez, which serial_update_e leaves alone; call it right after.
         *
         * Under PEC the edge Ez is set to zero. Under PMC it takes the interior update
         * with every H outside the domain read as zero: Hy(-1, j) and Hy(nx - 1, j),
         * Hx(i, -1) and Hx(i, ny - 1). Corners combine both.
         *
         * @param fields Fields to update.
         * @param coeffs Coefficients on the same grid as the fields.
         * @param boundary Outer boundary condition.
         */
        void serial_update_boundary(Fields2D &fields, const Coeffs2D &coeffs,
                                    Boundary boundary) noexcept;
    }
}

#endif
