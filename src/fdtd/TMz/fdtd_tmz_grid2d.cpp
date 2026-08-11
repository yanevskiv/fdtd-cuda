/*
 * C++ file for the 2D TMz grid.
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

// Class and function declarations.
#include "fdtd/TMz/fdtd_tmz_grid2d.h"

// Project constants for the speed of light.
#include "fdtd/fdtd_constants.h"

// Standard library for math, limits and errors.
#include <cmath>
#include <limits>
#include <stdexcept>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Validate the geometry and store it.
    Grid2D::Grid2D(std::size_t nx, std::size_t ny, FloatType dx, FloatType dy)
        : m_nx(nx)
        , m_ny(ny)
        , m_dx(dx)
        , m_dy(dy)
        , m_pitch(nx)
    {
        if (nx == 0 || ny == 0) {
            throw std::invalid_argument("Grid2D: cell counts must be at least 1");
        }
        if (!(dx > 0) || !(dy > 0)) {
            throw std::invalid_argument("Grid2D: cell sizes must be positive");
        }
        if (ny > std::numeric_limits<std::size_t>::max() / nx) {
            throw std::invalid_argument("Grid2D: nx * ny does not fit std::size_t");
        }
    }

    // Return the cells along x.
    std::size_t Grid2D::nx() const noexcept
    {
        return m_nx;
    }

    // Return the cells along y.
    std::size_t Grid2D::ny() const noexcept
    {
        return m_ny;
    }

    // Return the cell size along x.
    FloatType Grid2D::dx() const noexcept
    {
        return m_dx;
    }

    // Return the cell size along y.
    FloatType Grid2D::dy() const noexcept
    {
        return m_dy;
    }

    // Return the row pitch.
    std::size_t Grid2D::pitch() const noexcept
    {
        return m_pitch;
    }

    // Return the element count of an array over the grid.
    std::size_t Grid2D::size() const noexcept
    {
        return m_pitch * m_ny;
    }

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
