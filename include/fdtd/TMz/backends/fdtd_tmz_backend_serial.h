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
         * Pointers to the four coefficient arrays the serial routines read.
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
             * View four arrays over the same grid.
             *
             * @param ce1Data Ez decay factors.
             * @param ce2Data Ez curl factors.
             * @param ch1Data H decay factors.
             * @param ch2Data H curl factors.
             */
            CoeffsView(const FloatType *ce1Data, const FloatType *ce2Data, const FloatType *ch1Data,
                       const FloatType *ch2Data) noexcept;

            /** Ez decay factors. */
            const FloatType *ce1;

            /** Ez curl factors, before 1/dx or 1/dy. */
            const FloatType *ce2;

            /** H decay factors. */
            const FloatType *ch1;

            /** H curl factors, before 1/dx or 1/dy. */
            const FloatType *ch2;
        };

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
         * @param coeffs Coefficient arrays on the same grid as the fields.
         */
        void serial_update_h(Fields2D &fields, CoeffsView coeffs) noexcept;

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
         * @param coeffs Coefficient arrays on the same grid as the fields.
         */
        void serial_update_e(Fields2D &fields, CoeffsView coeffs) noexcept;

        /**
         * Update the edge Ez, which serial_update_e leaves alone; call it right after.
         *
         * Under PEC the edge Ez is set to zero. Under PMC it takes the interior update
         * with every H outside the domain read as zero: Hy(-1, j) and Hy(nx - 1, j),
         * Hx(i, -1) and Hx(i, ny - 1). Corners combine both.
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

            /** The backend's copy of Ce1. */
            AlignedBuffer<FloatType> m_ce1;

            /** The backend's copy of Ce2. */
            AlignedBuffer<FloatType> m_ce2;

            /** The backend's copy of Ch1. */
            AlignedBuffer<FloatType> m_ch1;

            /** The backend's copy of Ch2. */
            AlignedBuffer<FloatType> m_ch2;

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
