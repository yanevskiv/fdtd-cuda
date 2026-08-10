/*
 * C++ file for the 2D TMz split-field PML conductivity profiles.
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

// Function declarations.
#include "fdtd/TMz/fdtd_tmz_pml2d.h"

// Project constants for the vacuum impedance.
#include "fdtd/fdtd_constants.h"

// Standard library for math, sizes and errors.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Reject a layer that does not fit or a grading or reflection out of range.
    void check_pml(std::size_t nx, std::size_t ny, std::size_t thickness, double order,
                   double reflection)
    {
        if (2 * thickness >= nx || 2 * thickness >= ny) {
            throw std::invalid_argument("PML: 2 * thickness must be below nx and ny");
        }
        if (!(order > 0) || !std::isfinite(order)) {
            throw std::invalid_argument("PML: order must be positive and finite");
        }
        if (!(reflection > 0) || !(reflection < 1)) {
            throw std::invalid_argument("PML: reflection must be in (0, 1)");
        }
    }

    // Size sigma_max for a round trip through the layer damped to the target reflection.
    double pml_sigma_max(std::size_t thickness, double cellSize, double order,
                         double reflection) noexcept
    {
        return -(order + 1) * std::log(reflection) / (2 * ETA_0 * double(thickness) * cellSize);
    }

    // Measure how far a point lies past the inner edge of the nearer layer.
    double pml_depth(double position, std::size_t n, std::size_t thickness) noexcept
    {
        const double low = double(thickness) - position;
        const double high = position - double(n - 1 - thickness);
        return std::max({low, high, 0.0});
    }

    // Grade sigma_max by the relative depth raised to the order.
    double pml_sigma(double position, std::size_t n, std::size_t thickness, double order,
                     double sigmaMax) noexcept
    {
        if (thickness == 0) {
            return 0;
        }
        const double depth = std::min(pml_depth(position, n, thickness), double(thickness));
        return sigmaMax * std::pow(depth / double(thickness), order);
    }

    // Scale by mu0 / eps0.
    double pml_magnetic_sigma(double sigma) noexcept
    {
        return sigma * MU_0 / EPS_0;
    }
}
