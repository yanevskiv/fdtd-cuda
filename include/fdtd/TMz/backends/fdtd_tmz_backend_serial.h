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

#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_base.h"
#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_block.h"
#include "fdtd/TMz/backends/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"

#include <cstddef>
#include <span>
#include <string_view>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /**
             * Advance Hx and Hy over the whole grid by one step, as update_h_block does.
             *
             * @param fields Fields to update.
             * @param coeffs Coefficient arrays on the same grid as the fields.
             */
            void serial_update_h(Fields &fields, cpu::CoeffsView coeffs) noexcept;

            /**
             * Advance the Ez inside the PML over the whole grid by one step, as
             * update_e_interior_block does; with no PML that is every Ez off the edge.
             *
             * @param fields Fields to update.
             * @param coeffs Coefficient arrays on the same grid as the fields.
             * @param pml PML thickness in cells, zero for none.
             */
            void serial_update_e(Fields &fields, cpu::CoeffsView coeffs, std::size_t pml) noexcept;

            /**
             * Advance the Ez in the PML over the whole grid by one step as Ezx and Ezy, as
             * update_pml_block does.
             *
             * @param fields Fields to update.
             * @param coeffs Coefficient arrays on the same grid as the fields.
             * @param pml PML thickness in cells, with 2 pml below nx and ny; zero for none.
             */
            void serial_update_pml(Fields &fields, cpu::CoeffsView coeffs,
                                   std::size_t pml) noexcept;

            /**
             * Update the edge Ez of the whole grid, as update_boundary_block does; call it right
             * after serial_update_e and serial_update_pml.
             *
             * @param fields Fields to update.
             * @param coeffs Coefficient arrays on the same grid as the fields.
             * @param boundary Outer boundary condition.
             */
            void serial_update_boundary(Fields &fields, cpu::CoeffsView coeffs,
                                        Boundary boundary) noexcept;

            /** Single-threaded reference backend, the oracle for the others. */
            class SerialBackend final : public cpu::CpuBase
            {
            public:
                /**
                 * Create the backend with zeroed fields.
                 *
                 * @param grid Grid the backend works on.
                 */
                explicit SerialBackend(const Grid &grid);

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

            protected:
                /**
                 * Advance the fields and add the sources, the whole grid as one part.
                 *
                 * @param nsteps Number of time steps.
                 * @param points Distinct cells inside the grid.
                 * @param values One value per point per step, step-major.
                 */
                void advance(std::size_t nsteps, std::span<const PointInjection> points,
                             std::span<const FloatType> values) override;
            };
        }
    }
}

#endif
