/*
 * C++ header for the 2D TMz split-field PML conductivity profiles.
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

#ifndef FDTD_TMZ_PML2D_H
#define FDTD_TMZ_PML2D_H

#include <cstddef>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /**
         * Check the settings of a PML on a grid.
         *
         * @param nx Cells along x.
         * @param ny Cells along y.
         * @param thickness Layer thickness in cells, with 2 * thickness < nx and ny.
         * @param order Grading exponent m, positive and finite.
         * @param reflection Target reflection R0 at normal incidence, in (0, 1).
         * @throws std::invalid_argument When a setting is out of range.
         */
        void check_pml(std::size_t nx, std::size_t ny, std::size_t thickness, double order,
                       double reflection);

        /**
         * Electric conductivity at the outer edge of a graded PML in vacuum.
         *
         *     sigma_max = -(m + 1) ln R0 / (2 eta0 D),  D = thickness * cellSize,
         *
         * so that a wave crossing the layer and back at normal incidence is damped to R0.
         *
         * @param thickness Layer thickness in cells, positive.
         * @param cellSize Cell size across the layer in meters.
         * @param order Grading exponent m.
         * @param reflection Target reflection R0.
         * @return sigma_max in S/m.
         */
        double pml_sigma_max(std::size_t thickness, double cellSize, double order,
                             double reflection) noexcept;

        /**
         * Depth into the PML of a point on one axis, in cells.
         *
         * The axis has n Ez nodes at 0 .. n - 1, the PML walls behind nodes 0 and n - 1.
         * The layers end at nodes thickness and n - 1 - thickness, where the depth is
         * zero, and reach a depth of thickness at the walls. H points sit at half
         * positions, as i + 1/2.
         *
         * @param position Position along the axis in cells.
         * @param n Ez nodes along the axis.
         * @param thickness Layer thickness in cells.
         * @return Depth in cells, zero outside the layers.
         */
        double pml_depth(double position, std::size_t n, std::size_t thickness) noexcept;

        /**
         * Graded electric conductivity at a point on one axis.
         *
         *     sigma = sigma_max (depth / thickness)^m
         *
         * @param position Position along the axis in cells.
         * @param n Ez nodes along the axis.
         * @param thickness Layer thickness in cells; zero gives zero.
         * @param order Grading exponent m.
         * @param sigmaMax Conductivity at the walls, from pml_sigma_max().
         * @return sigma in S/m.
         */
        double pml_sigma(double position, std::size_t n, std::size_t thickness, double order,
                         double sigmaMax) noexcept;

        /**
         * Magnetic conductivity matched to an electric one, sigma* = sigma mu0 / eps0.
         *
         * With this match a vacuum PML has the impedance of vacuum, so a wave enters it
         * without reflection at any angle.
         *
         * @param sigma Electric conductivity in S/m.
         * @return sigma* in ohm/m.
         */
        double pml_magnetic_sigma(double sigma) noexcept;
    }
}

#endif
