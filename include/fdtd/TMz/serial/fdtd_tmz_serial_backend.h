/*
 * C++ header for the 2D TMz serial backend.
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

#ifndef FDTD_TMZ_SERIAL_BACKEND_H
#define FDTD_TMZ_SERIAL_BACKEND_H

#include "fdtd/TMz/fdtd_tmz_backend.h"
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
