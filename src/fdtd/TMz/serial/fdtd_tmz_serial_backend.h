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

#include <optional>

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
             * Advance the fields.
             *
             * @param nsteps Number of time steps.
             */
            void step(std::size_t nsteps) override;

        private:
            /** The backend's copy of the fields. */
            Fields2D m_fields;

            /** The backend's copy of the coefficients, empty until uploaded. */
            std::optional<Coeffs2D> m_coeffs;

            /** Outer boundary condition. */
            Boundary m_boundary = Boundary::Pec;
        };
    }
}

#endif
