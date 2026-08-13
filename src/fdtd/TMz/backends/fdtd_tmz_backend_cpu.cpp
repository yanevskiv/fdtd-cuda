/*
 * C++ file for the base of the 2D TMz CPU backends.
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

// Class declaration.
#include "fdtd/TMz/backends/fdtd_tmz_backend_cpu.h"

// Standard library for copies, sizes, errors, string building and buffers.
#include <algorithm>
#include <cstddef>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Helpers private to this file.
    namespace
    {
        // Check that two grids hold the same geometry.
        void require_same_grid(std::string_view backend, const Grid &expected, const Grid &actual)
        {
            if (expected.nx() != actual.nx() || expected.ny() != actual.ny() ||
                expected.pitch() != actual.pitch() || expected.dx() != actual.dx() ||
                expected.dy() != actual.dy()) {
                std::ostringstream message;
                message << backend;
                message << " backend: the data is on a different grid";
                throw std::invalid_argument(message.str());
            }
        }

        // Copy one array into another of the same size.
        void copy_field(std::span<const FloatType> source, std::span<FloatType> target)
        {
            std::copy(source.begin(), source.end(), target.begin());
        }
    }

    // Allocate the fields and coefficient arrays.
    CpuBackend::CpuBackend(const Grid &grid)
        : m_fields(grid)
        , m_coeff_ez_x1(grid.size())
        , m_coeff_ez_x2(grid.size())
        , m_coeff_ez_y1(grid.size())
        , m_coeff_ez_y2(grid.size())
        , m_coeff_hx_1(grid.size())
        , m_coeff_hx_2(grid.size())
        , m_coeff_hy_1(grid.size())
        , m_coeff_hy_2(grid.size())
    {
        // Empty.
    }

    // Return the grid.
    const Grid &CpuBackend::grid() const noexcept
    {
        return m_fields.grid();
    }

    // Copy the fields in.
    void CpuBackend::upload(const Fields &fields)
    {
        require_same_grid(name(), grid(), fields.grid());
        copy_field(fields.ez(), m_fields.ez());
        copy_field(fields.hx(), m_fields.hx());
        copy_field(fields.hy(), m_fields.hy());
        copy_field(fields.ezx(), m_fields.ezx());
        copy_field(fields.ezy(), m_fields.ezy());
    }

    // Copy the fields out.
    void CpuBackend::download(Fields &fields) const
    {
        require_same_grid(name(), grid(), fields.grid());
        copy_field(m_fields.ez(), fields.ez());
        copy_field(m_fields.hx(), fields.hx());
        copy_field(m_fields.hy(), fields.hy());
        copy_field(m_fields.ezx(), fields.ezx());
        copy_field(m_fields.ezy(), fields.ezy());
    }

    // Copy the coefficient arrays in.
    void CpuBackend::upload(const Coeffs &coeffs)
    {
        require_same_grid(name(), grid(), coeffs.grid());
        copy_field(coeffs.coeff_ez_x1(), {m_coeff_ez_x1.data(), m_coeff_ez_x1.size()});
        copy_field(coeffs.coeff_ez_x2(), {m_coeff_ez_x2.data(), m_coeff_ez_x2.size()});
        copy_field(coeffs.coeff_ez_y1(), {m_coeff_ez_y1.data(), m_coeff_ez_y1.size()});
        copy_field(coeffs.coeff_ez_y2(), {m_coeff_ez_y2.data(), m_coeff_ez_y2.size()});
        copy_field(coeffs.coeff_hx_1(), {m_coeff_hx_1.data(), m_coeff_hx_1.size()});
        copy_field(coeffs.coeff_hx_2(), {m_coeff_hx_2.data(), m_coeff_hx_2.size()});
        copy_field(coeffs.coeff_hy_1(), {m_coeff_hy_1.data(), m_coeff_hy_1.size()});
        copy_field(coeffs.coeff_hy_2(), {m_coeff_hy_2.data(), m_coeff_hy_2.size()});
        m_pml_thickness = coeffs.pml_thickness();
        m_has_coeffs = true;
    }

    // Store the boundary condition.
    void CpuBackend::set_boundary(Boundary boundary)
    {
        m_boundary = boundary;
    }

    // Return the boundary condition.
    Boundary CpuBackend::boundary() const noexcept
    {
        return m_boundary;
    }

    // Return the number of probes.
    std::size_t CpuBackend::probe_count() const noexcept
    {
        return m_probes.size();
    }

    // Return the number of recorded steps.
    std::size_t CpuBackend::probe_steps() const noexcept
    {
        return m_probes.empty() ? 0 : m_records.size() / m_probes.size();
    }

    // Copy the records out.
    std::vector<FloatType> CpuBackend::fetch_probes() const
    {
        return m_records;
    }

    // Keep the probe cells and clear the records.
    void CpuBackend::reset_probes(std::span<const PointProbe> points)
    {
        m_probes.assign(points.begin(), points.end());
        m_records.clear();
    }

    // Throw unless coefficients were uploaded.
    void CpuBackend::require_coeffs() const
    {
        if (!m_has_coeffs) {
            std::ostringstream message;
            message << name();
            message << " backend: step before the coefficients were uploaded";
            throw std::logic_error(message.str());
        }
    }

    // Return the fields.
    Fields &CpuBackend::fields() noexcept
    {
        return m_fields;
    }

    // View the coefficient arrays.
    CoeffsView CpuBackend::coeffs() const noexcept
    {
        return {m_coeff_ez_x1.data(), m_coeff_ez_x2.data(), m_coeff_ez_y1.data(),
                m_coeff_ez_y2.data(), m_coeff_hx_1.data(),  m_coeff_hx_2.data(),
                m_coeff_hy_1.data(),  m_coeff_hy_2.data()};
    }

    // Return the PML thickness under Boundary::Pml, zero otherwise.
    std::size_t CpuBackend::active_pml() const noexcept
    {
        return m_boundary == Boundary::Pml ? m_pml_thickness : 0;
    }

    // Return the probe cells.
    std::span<const PointProbe> CpuBackend::probes() const noexcept
    {
        return m_probes;
    }

    // Grow the records by nsteps steps of zeros and point at the first new one.
    FloatType *CpuBackend::add_records(std::size_t nsteps)
    {
        const std::size_t first = m_records.size();
        m_records.resize(first + nsteps * m_probes.size());
        return m_records.data() + first;
    }
}
