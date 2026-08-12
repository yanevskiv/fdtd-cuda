/*
 * C++ header for the 2D TMz material map.
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

#ifndef FDTD_TMZ_MATERIAL_H
#define FDTD_TMZ_MATERIAL_H

#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <span>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /**
         * Per-cell material parameters over a grid, vacuum by default.
         *
         * Each array has grid().size() elements, indexed with Grid::index. Permittivity
         * and permeability are relative; the conductivities are in SI units.
         *
         * Besides the isotropic media, the material holds the directional conductivities of
         * a split-field PML along the edges, zero until set_pml() grades them. They are the
         * vacuum profiles: the coefficients scale them by eps_r and mu_r, so the media and
         * the PML can be set in any order, and the fill functions leave them alone.
         */
        class Material
        {
        public:
            /** Parameters of a homogeneous medium. */
            struct Medium
            {
                /** Relative permittivity, positive. */
                FloatType eps_r = 1;

                /** Relative permeability, positive. */
                FloatType mu_r = 1;

                /** Electric conductivity in S/m, non-negative. */
                FloatType sigma_e = 0;

                /** Magnetic conductivity in ohm/m, non-negative. */
                FloatType sigma_m = 0;
            };

            /**
             * Fill a grid with vacuum.
             *
             * @param grid Grid the material lives on.
             */
            explicit Material(const Grid &grid);

            /**
             * Grid the material lives on.
             *
             * @return The grid.
             */
            const Grid &grid() const noexcept;

            /**
             * Medium of one cell.
             *
             * @param i Cell along x.
             * @param j Cell along y.
             * @return The medium.
             * @throws std::out_of_range When the cell is outside the grid.
             */
            Medium at(std::size_t i, std::size_t j) const;

            /**
             * Set every cell to a medium.
             *
             * @param medium Medium to fill with.
             * @throws std::invalid_argument When the medium is not physical.
             */
            void fill(const Medium &medium);

            /**
             * Set the cells i0 <= i < i1, j0 <= j < j1 to a medium, clipped to the grid.
             *
             * @param i0 First cell along x.
             * @param j0 First cell along y.
             * @param i1 One past the last cell along x.
             * @param j1 One past the last cell along y.
             * @param medium Medium to fill with.
             * @throws std::invalid_argument When the medium is not physical.
             */
            void fill_rect(std::size_t i0, std::size_t j0, std::size_t i1, std::size_t j1,
                           const Medium &medium);

            /**
             * Set the cells whose indices lie within a circle to a medium, clipped to the grid.
             *
             * Cell (i, j) is inside when (i - ci)^2 + (j - cj)^2 <= radius^2, so the center
             * may lie off the grid.
             *
             * @param ci Center along x in cells.
             * @param cj Center along y in cells.
             * @param radius Radius in cells, non-negative.
             * @param medium Medium to fill with.
             * @throws std::invalid_argument When the medium is not physical or the circle
             *         is not finite and non-negative.
             */
            void fill_circle(double ci, double cj, double radius, const Medium &medium);

            /**
             * Relative permittivity.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> eps_r() const noexcept;

            /**
             * Relative permeability.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> mu_r() const noexcept;

            /**
             * Electric conductivity in S/m.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> sigma_e() const noexcept;

            /**
             * Magnetic conductivity in ohm/m.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> sigma_m() const noexcept;

            /**
             * Grade the PML conductivities along the edges, or clear them.
             *
             * The axis along x has nx Ez nodes 0 .. nx - 1, with the PML walls behind nodes
             * 0 and nx - 1; the layers end at nodes thickness and nx - 1 - thickness. At a
             * depth d into a layer, in cells,
             *
             *     sigma(d) = sigma_max (d / thickness)^order,
             *     sigma_max = -(order + 1) ln reflection / (2 eta0 thickness dx),
             *
             * so a wave crossing the layer and back at normal incidence is damped to the
             * reflection. The magnetic profiles are matched, sigma* = sigma mu0 / eps0, and
             * sampled half a cell further, where Hy and Hx sit. Likewise along y, with dy.
             *
             * @param thickness Layer thickness in cells, with 2 * thickness < nx and ny;
             *        zero clears the profiles and ignores order and reflection.
             * @param order Grading exponent, positive and finite.
             * @param reflection Target reflection at normal incidence, in (0, 1).
             * @throws std::invalid_argument When a setting is out of range; the material is
             *         then unchanged.
             */
            void set_pml(std::size_t thickness, double order, double reflection);

            /**
             * Thickness of the PML.
             *
             * @return Layer thickness in cells, zero without a PML.
             */
            std::size_t pml_thickness() const noexcept;

            /**
             * PML electric conductivity along x in S/m, at the Ez node of each cell (i).
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> sigma_e_x() const noexcept;

            /**
             * PML electric conductivity along y in S/m, at the Ez node of each cell (j).
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> sigma_e_y() const noexcept;

            /**
             * PML magnetic conductivity along x in ohm/m, at the Hy node of each cell (i + 1/2).
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> sigma_m_x() const noexcept;

            /**
             * PML magnetic conductivity along y in ohm/m, at the Hx node of each cell (j + 1/2).
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> sigma_m_y() const noexcept;

        private:
            /**
             * Set one cell to a medium, unchecked.
             *
             * @param k Index of the cell.
             * @param medium Medium to set.
             */
            void set(std::size_t k, const Medium &medium) noexcept;

            /** Grid the material lives on. */
            Grid m_grid;

            /** Relative permittivity per cell. */
            std::vector<FloatType> m_eps_r;

            /** Relative permeability per cell. */
            std::vector<FloatType> m_mu_r;

            /** Electric conductivity per cell in S/m. */
            std::vector<FloatType> m_sigma_e;

            /** Magnetic conductivity per cell in ohm/m. */
            std::vector<FloatType> m_sigma_m;

            /** PML thickness in cells, zero without a PML. */
            std::size_t m_pml_thickness = 0;

            /** PML electric conductivity along x per cell in S/m. */
            std::vector<FloatType> m_sigma_e_x;

            /** PML electric conductivity along y per cell in S/m. */
            std::vector<FloatType> m_sigma_e_y;

            /** PML magnetic conductivity along x per cell in ohm/m. */
            std::vector<FloatType> m_sigma_m_x;

            /** PML magnetic conductivity along y per cell in ohm/m. */
            std::vector<FloatType> m_sigma_m_y;
        };
    }
}

#endif
