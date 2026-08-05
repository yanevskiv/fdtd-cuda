/*
 * C++ file for the 2D TMz Courant time step.
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

// Function declaration.
#include "fdtd/TMz/fdtd_tmz_courant.h"

// Project constants for the speed of light.
#include "fdtd/fdtd_constants.h"

// Standard library for math and errors.
#include <cmath>
#include <stdexcept>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Check the Courant number and scale the 2D stability limit by it.
    double courant_dt(const Grid2D &grid, double courant)
    {
        if (!(courant > 0) || !(courant <= 1)) {
            throw std::invalid_argument("courant_dt: Courant number must be in (0, 1]");
        }
        const double invDx = 1.0 / double(grid.dx());
        const double invDy = 1.0 / double(grid.dy());
        return courant / (C_0 * std::sqrt(invDx * invDx + invDy * invDy));
    }
}
