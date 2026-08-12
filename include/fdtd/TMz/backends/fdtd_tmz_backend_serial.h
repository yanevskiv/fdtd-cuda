/*
 * C++ header for the 2D TMz serial backend and its whole-grid update routines.
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

#include "fdtd/TMz/backends/fdtd_tmz_block2d.h"
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
         * Advance Hx and Hy over the whole grid by one step, as update_h_block does.
         *
         * @param fields Fields to update.
         * @param coeffs Coefficient arrays on the same grid as the fields.
         */
        void serial_update_h(Fields2D &fields, CoeffsView coeffs) noexcept;

        /**
         * Advance the Ez inside the PML over the whole grid by one step, as
         * update_e_interior_block does; with no PML that is every Ez off the edge.
         *
         * @param fields Fields to update.
         * @param coeffs Coefficient arrays on the same grid as the fields.
         * @param pml PML thickness in cells, zero for none.
         */
        void serial_update_e(Fields2D &fields, CoeffsView coeffs, std::size_t pml) noexcept;

        /**
         * Advance the Ez in the PML over the whole grid by one step as Ezx and Ezy, as
         * update_pml_block does.
         *
         * @param fields Fields to update.
         * @param coeffs Coefficient arrays on the same grid as the fields.
         * @param pml PML thickness in cells, with 2 pml below nx and ny; zero for none.
         */
        void serial_update_pml(Fields2D &fields, CoeffsView coeffs, std::size_t pml) noexcept;

        /**
         * Update the edge Ez of the whole grid, as update_boundary_block does; call it right
         * after serial_update_e and serial_update_pml.
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

            /** PML thickness the uploaded coefficients carry, used under Boundary::Pml. */
            std::size_t m_pml_thickness = 0;

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
