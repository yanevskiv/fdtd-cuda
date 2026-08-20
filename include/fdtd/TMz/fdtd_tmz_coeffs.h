/*
 * C++ header for the 2D TMz update coefficients.
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

#ifndef FDTD_TMZ_COEFFS_H
#define FDTD_TMZ_COEFFS_H

#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_material.h"
#include "fdtd/fdtd_aligned_buffer.h"
#include "fdtd/fdtd_float.h"

#include <span>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /**
         * Per-cell update coefficients of a material for a time step, by field and direction.
         *
         * Each pair is a decay factor and a curl factor for one loss term,
         *
         *     C1 = (1 - a) / (1 + a),  C2 = (dt / eps) / (1 + a),  a = sigma dt / (2 eps),
         *
         * with eps = eps_r EPS_0 for Ez and mu_r MU_0 in its place for Hx and Hy. The
         * conductivity is the isotropic one of the material plus its PML profile along
         * one direction, scaled by eps_r or mu_r:
         *
         *     Ez along x: sigma_e + eps_r sigma_e_x,  Ez along y: sigma_e + eps_r sigma_e_y,
         *     Hx:         sigma_m + mu_r sigma_m_y,   Hy:         sigma_m + mu_r sigma_m_x.
         *
         * Without a PML the x and y pairs of Ez are equal, and Hx and Hy are equal. The curl
         * factors leave out the cell size: the update applies 1/dx and 1/dy to the
         * differences. Each array has grid().size() elements, indexed with Grid::index.
         * Computed in double.
         */
        class Coeffs
        {
        public:
            /**
             * Compute the coefficients of a material.
             *
             * @param material Material to compute from; its grid becomes grid().
             * @param dt Time step in seconds, positive and finite.
             * @throws std::invalid_argument When dt is not positive and finite.
             */
            Coeffs(const Material &material, double dt);

            /** Coefficients are not copyable; a backend copies the arrays it needs. */
            Coeffs(const Coeffs &) = delete;

            /**
             * Coefficients are not copyable.
             *
             * @return These coefficients.
             */
            Coeffs &operator=(const Coeffs &) = delete;

            /** Take over another set's arrays. */
            Coeffs(Coeffs &&) noexcept = default;

            /**
             * Take over another set's arrays.
             *
             * @return These coefficients.
             */
            Coeffs &operator=(Coeffs &&) noexcept = default;

            /** Release the arrays. */
            ~Coeffs() = default;

            /**
             * Grid the coefficients live on.
             *
             * @return The grid.
             */
            const Grid &grid() const noexcept;

            /**
             * Time step the coefficients were computed for.
             *
             * @return Time step in seconds.
             */
            double dt() const noexcept;

            /**
             * Thickness of the PML the material carried.
             *
             * @return Material::pml_thickness() of the material, zero without a PML.
             */
            std::size_t pml_thickness() const noexcept;

            /**
             * Ez decay factor along x.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> coeff_ez_x1() const noexcept;

            /**
             * Ez curl factor along x, before 1/dx.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> coeff_ez_x2() const noexcept;

            /**
             * Ez decay factor along y.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> coeff_ez_y1() const noexcept;

            /**
             * Ez curl factor along y, before 1/dy.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> coeff_ez_y2() const noexcept;

            /**
             * Hx decay factor.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> coeff_hx_1() const noexcept;

            /**
             * Hx curl factor, before 1/dy.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> coeff_hx_2() const noexcept;

            /**
             * Hy decay factor.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> coeff_hy_1() const noexcept;

            /**
             * Hy curl factor, before 1/dx.
             *
             * @return Read-only view of the array.
             */
            std::span<const FloatType> coeff_hy_2() const noexcept;

        private:
            /** Grid the coefficients live on. */
            Grid m_grid;

            /** Time step in seconds. */
            double m_dt;

            /** PML thickness of the material in cells. */
            std::size_t m_pml_thickness;

            /** Ez decay factor along x, per cell. */
            AlignedBuffer<FloatType> m_coeff_ez_x1;

            /** Ez curl factor along x, before 1/dx, per cell. */
            AlignedBuffer<FloatType> m_coeff_ez_x2;

            /** Ez decay factor along y, per cell. */
            AlignedBuffer<FloatType> m_coeff_ez_y1;

            /** Ez curl factor along y, before 1/dy, per cell. */
            AlignedBuffer<FloatType> m_coeff_ez_y2;

            /** Hx decay factor, per cell. */
            AlignedBuffer<FloatType> m_coeff_hx_1;

            /** Hx curl factor, before 1/dy, per cell. */
            AlignedBuffer<FloatType> m_coeff_hx_2;

            /** Hy decay factor, per cell. */
            AlignedBuffer<FloatType> m_coeff_hy_1;

            /** Hy curl factor, before 1/dx, per cell. */
            AlignedBuffer<FloatType> m_coeff_hy_2;
        };
    }
}

#endif
