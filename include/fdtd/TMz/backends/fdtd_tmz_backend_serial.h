/*
 * C++ header for the 2D TMz serial backend and its update routines.
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

#ifndef FDTD_TMZ_BACKEND_SERIAL_H
#define FDTD_TMZ_BACKEND_SERIAL_H

#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_coeffs2d.h"
#include "fdtd/TMz/fdtd_tmz_fields2d.h"
#include "fdtd/fdtd_aligned_buffer.h"

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
         * Pointers to the eight coefficient arrays the serial routines read.
         *
         * The routines take these rather than a Coeffs2D so the backend can keep its own
         * arrays, as a GPU backend keeps device arrays. A Coeffs2D converts implicitly.
         */
        struct CoeffsView
        {
            /**
             * View the arrays of a coefficient set, which must outlive the view.
             *
             * @param coeffs Coefficients to view.
             */
            CoeffsView(const Coeffs2D &coeffs) noexcept; // NOLINT(google-explicit-constructor)

            /**
             * View eight arrays over the same grid, laid out as in Coeffs2D.
             *
             * @param ezX1 Ez decay factors along x.
             * @param ezX2 Ez curl factors along x.
             * @param ezY1 Ez decay factors along y.
             * @param ezY2 Ez curl factors along y.
             * @param hx1 Hx decay factors.
             * @param hx2 Hx curl factors.
             * @param hy1 Hy decay factors.
             * @param hy2 Hy curl factors.
             */
            CoeffsView(const FloatType *ezX1, const FloatType *ezX2, const FloatType *ezY1,
                       const FloatType *ezY2, const FloatType *hx1, const FloatType *hx2,
                       const FloatType *hy1, const FloatType *hy2) noexcept;

            /** Ez decay factors along x. */
            const FloatType *coeff_ez_x1;

            /** Ez curl factors along x, before 1/dx. */
            const FloatType *coeff_ez_x2;

            /** Ez decay factors along y. */
            const FloatType *coeff_ez_y1;

            /** Ez curl factors along y, before 1/dy. */
            const FloatType *coeff_ez_y2;

            /** Hx decay factors. */
            const FloatType *coeff_hx_1;

            /** Hx curl factors, before 1/dy. */
            const FloatType *coeff_hx_2;

            /** Hy decay factors. */
            const FloatType *coeff_hy_1;

            /** Hy curl factors, before 1/dx. */
            const FloatType *coeff_hy_2;
        };

        /**
         * Advance Hx and Hy by one step from the differences of Ez.
         *
         * With Hx(i, j) at (i, j + 1/2) and Hy(i, j) at (i + 1/2, j):
         *
         *     Hx(i, j) = Chx1 Hx(i, j) - Chx2 / dy (Ez(i, j + 1) - Ez(i, j)),  j < ny - 1,
         *     Hy(i, j) = Chy1 Hy(i, j) + Chy2 / dx (Ez(i + 1, j) - Ez(i, j)),  i < nx - 1,
         *
         * with the coefficients of cell (i, j). Hx on the last row and Hy on the last
         * column lie outside the domain and are left untouched. Inner loop along x.
         *
         * @param fields Fields to update.
         * @param coeffs Coefficient arrays on the same grid as the fields.
         */
        void serial_update_h(Fields2D &fields, CoeffsView coeffs) noexcept;

        /**
         * Advance Ez by one step from the curl of H.
         *
         *     Ez(i, j) = Cex1 Ez(i, j) + Cex2 ((Hy(i, j) - Hy(i - 1, j)) / dx
         *                                    - (Hx(i, j) - Hx(i, j - 1)) / dy)
         *
         * over the interior 0 < i < nx - 1, 0 < j < ny - 1, with the x coefficients, which
         * equal the y ones outside a PML. The edge Ez is left to the outer boundary. Inner
         * loop along x.
         *
         * @param fields Fields to update.
         * @param coeffs Coefficient arrays on the same grid as the fields.
         */
        void serial_update_e(Fields2D &fields, CoeffsView coeffs) noexcept;

        /**
         * Update the edge Ez, which serial_update_e leaves alone; call it right after.
         *
         * Under PEC and PML the edge Ez is set to zero. Under PMC it takes the interior update,
         * with the x coefficients, and every H outside the domain read as zero: Hy(-1, j) and Hy(nx
         * - 1, j), Hx(i, -1) and Hx(i, ny - 1). Corners combine both.
         *
         * @param fields Fields to update.
         * @param coeffs Coefficient arrays on the same grid as the fields.
         * @param boundary Outer boundary condition.
         */
        void serial_update_boundary(Fields2D &fields, CoeffsView coeffs,
                                    Boundary boundary) noexcept;

        /** Single-threaded reference backend, the oracle for the others. */
        class SerialBackend final : public Backend
        {
        public:
            /**
             * Create the backend with zeroed fields.
             *
             * @param grid Grid the backend works on.
             */
            explicit SerialBackend(const Grid2D &grid);

            /**
             * Short lowercase name of the backend.
             *
             * @return "serial".
             */
            std::string_view name() const noexcept override;

            /**
             * Type of the backend.
             *
             * @return BackendType::Serial.
             */
            BackendType type() const noexcept override;

            /**
             * Grid the backend was made for.
             *
             * @return The grid.
             */
            const Grid2D &grid() const noexcept override;

            /**
             * Copy fields into the backend.
             *
             * @param fields Fields on the backend's grid.
             */
            void upload(const Fields2D &fields) override;

            /**
             * Copy the backend's fields out.
             *
             * @param fields Destination on the backend's grid.
             */
            void download(Fields2D &fields) const override;

            /**
             * Copy update coefficients into the backend.
             *
             * @param coeffs Coefficients on the backend's grid.
             */
            void upload(const Coeffs2D &coeffs) override;

            /**
             * Set the outer boundary condition.
             *
             * @param boundary Outer boundary condition.
             */
            void set_boundary(Boundary boundary) override;

            /**
             * Outer boundary condition.
             *
             * @return The boundary.
             */
            Boundary boundary() const noexcept override;

            /**
             * Number of probes set.
             *
             * @return The probe count.
             */
            std::size_t probe_count() const noexcept override;

            /**
             * Number of steps recorded since the probes were set.
             *
             * @return The recorded step count.
             */
            std::size_t probe_steps() const noexcept override;

            /**
             * Copy the probe records out, step-major.
             *
             * @return The records.
             */
            std::vector<FloatType> fetch_probes() const override;

        protected:
            /**
             * Advance the fields and add the sources.
             *
             * @param nsteps Number of time steps.
             * @param points Distinct cells inside the grid.
             * @param values One value per point per step, step-major.
             */
            void advance(std::size_t nsteps, std::span<const PointInjection> points,
                         std::span<const FloatType> values) override;

            /**
             * Replace the probes and clear the records.
             *
             * @param points Cells inside the grid.
             */
            void reset_probes(std::span<const PointProbe> points) override;

        private:
            /** The backend's copy of the fields. */
            Fields2D m_fields;

            /** Whether coefficients were uploaded. */
            bool m_has_coeffs = false;

            /** The backend's copy of the Ez decay factors along x. */
            AlignedBuffer<FloatType> m_coeff_ez_x1;

            /** The backend's copy of the Ez curl factors along x. */
            AlignedBuffer<FloatType> m_coeff_ez_x2;

            /** The backend's copy of the Ez decay factors along y. */
            AlignedBuffer<FloatType> m_coeff_ez_y1;

            /** The backend's copy of the Ez curl factors along y. */
            AlignedBuffer<FloatType> m_coeff_ez_y2;

            /** The backend's copy of the Hx decay factors. */
            AlignedBuffer<FloatType> m_coeff_hx_1;

            /** The backend's copy of the Hx curl factors. */
            AlignedBuffer<FloatType> m_coeff_hx_2;

            /** The backend's copy of the Hy decay factors. */
            AlignedBuffer<FloatType> m_coeff_hy_1;

            /** The backend's copy of the Hy curl factors. */
            AlignedBuffer<FloatType> m_coeff_hy_2;

            /** Outer boundary condition. */
            Boundary m_boundary = Boundary::Pec;

            /** Probe cells. */
            std::vector<PointProbe> m_probes;

            /** Probe records, step-major. */
            std::vector<FloatType> m_records;
        };
    }
}

#endif
