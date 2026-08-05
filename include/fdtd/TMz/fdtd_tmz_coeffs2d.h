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

#ifndef FDTD_TMZ_COEFFS2D_H
#define FDTD_TMZ_COEFFS2D_H

#include "fdtd/TMz/fdtd_tmz_grid2d.h"
#include "fdtd/TMz/fdtd_tmz_material2d.h"
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
         * Per-cell update coefficients of a material for a time step, as four aligned arrays.
         *
         * With the semi-implicit loss terms a = sigma_e dt / (2 eps) and
         * b = sigma_m dt / (2 mu), where eps = eps_r EPS_0 and mu = mu_r MU_0:
         *
         *     Ce1 = (1 - a) / (1 + a),  Ce2 = (dt / eps) / (1 + a),
         *     Ch1 = (1 - b) / (1 + b),  Ch2 = (dt / mu) / (1 + b).
         *
         * Unlike fdtd-openmp, Ce2 and Ch2 leave out the cell size: the update applies
         * 1/dx and 1/dy to the differences, so one Ch2 serves both Hx and Hy. Each array
         * has grid().size() elements, indexed with Grid2D::index. Computed in double.
         */
        class Coeffs2D
        {
        public:
            /**
             * Compute the coefficients of a material.
             *
             * @param material Material to compute from; its grid becomes grid().
             * @param dt Time step in seconds, positive and finite.
             * @throws std::invalid_argument When dt is not positive and finite.
             */
            Coeffs2D(const Material2D &material, double dt);

            /**
             * Copy the coefficients into new arrays; a backend keeps its own copy.
             *
             * @param other Coefficients to copy.
             */
            Coeffs2D(const Coeffs2D &other);

            /**
             * Replace the coefficients with a copy of another set.
             *
             * @param other Coefficients to copy.
             * @return These coefficients.
             */
            Coeffs2D &operator=(const Coeffs2D &other);

            /** Take over another set's arrays. */
            Coeffs2D(Coeffs2D &&) noexcept = default;

            /**
             * Take over another set's arrays.
             *
             * @return These coefficients.
             */
            Coeffs2D &operator=(Coeffs2D &&) noexcept = default;

            /** Release the arrays. */
            ~Coeffs2D() = default;

            /**
             * Grid the coefficients live on.
             *
             * @return The grid.
             */
            const Grid2D &grid() const noexcept;

            /**
             * Time step the coefficients were computed for.
             *
             * @return Time step in seconds.
             */
            double dt() const noexcept;

            /**
             * Ez decay factor.
             *
             * @return Read-only view of the Ce1 array.
             */
            std::span<const FloatType> ce1() const noexcept;

            /**
             * Ez curl factor, before 1/dx or 1/dy.
             *
             * @return Read-only view of the Ce2 array.
             */
            std::span<const FloatType> ce2() const noexcept;

            /**
             * Hx and Hy decay factor.
             *
             * @return Read-only view of the Ch1 array.
             */
            std::span<const FloatType> ch1() const noexcept;

            /**
             * Hx and Hy curl factor, before 1/dx or 1/dy.
             *
             * @return Read-only view of the Ch2 array.
             */
            std::span<const FloatType> ch2() const noexcept;

        private:
            /** Grid the coefficients live on. */
            Grid2D m_grid;

            /** Time step in seconds. */
            double m_dt;

            /** Ez decay factor per cell. */
            AlignedBuffer<FloatType> m_ce1;

            /** Ez curl factor per cell. */
            AlignedBuffer<FloatType> m_ce2;

            /** H decay factor per cell. */
            AlignedBuffer<FloatType> m_ch1;

            /** H curl factor per cell. */
            AlignedBuffer<FloatType> m_ch2;
        };
    }
}

#endif
