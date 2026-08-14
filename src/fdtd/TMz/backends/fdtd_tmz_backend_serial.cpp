/*
 * C++ file for the 2D TMz serial backend and its update routines.
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

// Class and routine declarations.
#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"

// The partition of the grid and the stepper over it.
#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_partition.h"
#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_stepper.h"

// Standard library for sizes and views.
#include <cstddef>
#include <span>
#include <string_view>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Update H over the whole grid.
    void serial_update_h(Fields &fields, cpu::CoeffsView coeffs) noexcept
    {
        cpu::update_h_block(fields, coeffs, cpu::whole_block(fields.grid()));
    }

    // Update the interior Ez over the whole grid, inside the PML when there is one.
    void serial_update_e(Fields &fields, cpu::CoeffsView coeffs, std::size_t pml) noexcept
    {
        cpu::update_e_interior_block(fields, coeffs, cpu::whole_block(fields.grid()), pml);
    }

    // Update Ezx, Ezy and Ez over the PML slabs.
    void serial_update_pml(Fields &fields, cpu::CoeffsView coeffs, std::size_t pml) noexcept
    {
        cpu::update_pml_block(fields, coeffs, cpu::whole_block(fields.grid()), pml);
    }

    // Update the edge Ez of the whole grid.
    void serial_update_boundary(Fields &fields, cpu::CoeffsView coeffs, Boundary boundary) noexcept
    {
        cpu::update_boundary_block(fields, coeffs, cpu::whole_block(fields.grid()), boundary);
    }

    // Create the shared storage.
    SerialBackend::SerialBackend(const Grid &grid)
        : cpu::CpuBase(grid)
    {
        // Empty.
    }

    // Return the backend name.
    std::string_view SerialBackend::name() const noexcept
    {
        return backend_name(BackendType::Serial);
    }

    // Return the backend type.
    BackendType SerialBackend::type() const noexcept
    {
        return BackendType::Serial;
    }

    // Step the whole grid as one part, the H phase then the E phase of every step.
    void SerialBackend::advance(std::size_t nsteps, std::span<const PointInjection> points,
                                std::span<const FloatType> values)
    {
        require_coeffs();
        FloatType *records = add_records(nsteps);
        const cpu::Partition partition = cpu::Partition::whole(grid());
        const cpu::Stepper stepper(fields(), coeffs(), active_pml(), boundary(), partition, points,
                                   values, probes(), records);
        for (std::size_t n = 0; n < nsteps; ++n) {
            stepper.h_phase(0, n);
            stepper.e_phase(0, n);
        }
    }
}
