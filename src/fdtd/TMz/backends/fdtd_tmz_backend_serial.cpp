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

// Standard library for copies, sizes, errors and buffers.
#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Update H over the whole grid.
    void serial_update_h(Fields2D &fields, CoeffsView coeffs) noexcept
    {
        update_h_block(fields, coeffs, whole_block(fields.grid()));
    }

    // Update the interior Ez over the whole grid, inside the PML when there is one.
    void serial_update_e(Fields2D &fields, CoeffsView coeffs, std::size_t pml) noexcept
    {
        update_e_interior_block(fields, coeffs, whole_block(fields.grid()), pml);
    }

    // Update Ezx, Ezy and Ez over the PML slabs.
    void serial_update_pml(Fields2D &fields, CoeffsView coeffs, std::size_t pml) noexcept
    {
        update_pml_block(fields, coeffs, whole_block(fields.grid()), pml);
    }

    // Update the edge Ez of the whole grid.
    void serial_update_boundary(Fields2D &fields, CoeffsView coeffs, Boundary boundary) noexcept
    {
        update_boundary_block(fields, coeffs, whole_block(fields.grid()), boundary);
    }

    // Check that two grids hold the same geometry.
    static void require_same_grid(const Grid2D &expected, const Grid2D &actual)
    {
        if (expected.nx() != actual.nx() || expected.ny() != actual.ny() ||
            expected.pitch() != actual.pitch() || expected.dx() != actual.dx() ||
            expected.dy() != actual.dy()) {
            throw std::invalid_argument("SerialBackend: the data is on a different grid");
        }
    }

    // Copy one array into another of the same size.
    static void copy_field(std::span<const FloatType> source, std::span<FloatType> target)
    {
        std::copy(source.begin(), source.end(), target.begin());
    }

    // Add one step's values to the points of the H components, or of Ez.
    static void add_sources(Fields2D &fields, std::span<const PointInjection> points,
                            const FloatType *values, bool magnetic)
    {
        for (std::size_t k = 0; k < points.size(); ++k) {
            const PointInjection &point = points[k];
            switch (point.component) {
                case Component::Ez:
                    if (!magnetic) {
                        fields.ez()[point.index] += values[k];
                    }
                    break;
                case Component::Hx:
                    if (magnetic) {
                        fields.hx()[point.index] += values[k];
                    }
                    break;
                case Component::Hy:
                    if (magnetic) {
                        fields.hy()[point.index] += values[k];
                    }
                    break;
            }
        }
    }

    // Return the value of one component at a cell.
    static FloatType read_point(const Fields2D &fields, const PointProbe &point)
    {
        switch (point.component) {
            case Component::Hx:
                return fields.hx()[point.index];
            case Component::Hy:
                return fields.hy()[point.index];
            case Component::Ez:
                break;
        }
        return fields.ez()[point.index];
    }

    // Allocate the backend's fields and coefficient arrays.
    SerialBackend::SerialBackend(const Grid2D &grid)
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

    // Return the grid.
    const Grid2D &SerialBackend::grid() const noexcept
    {
        return m_fields.grid();
    }

    // Copy the fields in.
    void SerialBackend::upload(const Fields2D &fields)
    {
        require_same_grid(grid(), fields.grid());
        copy_field(fields.ez(), m_fields.ez());
        copy_field(fields.hx(), m_fields.hx());
        copy_field(fields.hy(), m_fields.hy());
        copy_field(fields.ezx(), m_fields.ezx());
        copy_field(fields.ezy(), m_fields.ezy());
    }

    // Copy the fields out.
    void SerialBackend::download(Fields2D &fields) const
    {
        require_same_grid(grid(), fields.grid());
        copy_field(m_fields.ez(), fields.ez());
        copy_field(m_fields.hx(), fields.hx());
        copy_field(m_fields.hy(), fields.hy());
        copy_field(m_fields.ezx(), fields.ezx());
        copy_field(m_fields.ezy(), fields.ezy());
    }

    // Copy the coefficient arrays in.
    void SerialBackend::upload(const Coeffs2D &coeffs)
    {
        require_same_grid(grid(), coeffs.grid());
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
    void SerialBackend::set_boundary(Boundary boundary)
    {
        m_boundary = boundary;
    }

    // Return the boundary condition.
    Boundary SerialBackend::boundary() const noexcept
    {
        return m_boundary;
    }

    // Return the number of probes.
    std::size_t SerialBackend::probe_count() const noexcept
    {
        return m_probes.size();
    }

    // Return the number of recorded steps.
    std::size_t SerialBackend::probe_steps() const noexcept
    {
        return m_probes.empty() ? 0 : m_records.size() / m_probes.size();
    }

    // Copy the records out.
    std::vector<FloatType> SerialBackend::fetch_probes() const
    {
        return m_records;
    }

    // Keep the probe cells and clear the records.
    void SerialBackend::reset_probes(std::span<const PointProbe> points)
    {
        m_probes.assign(points.begin(), points.end());
        m_records.clear();
    }

    // Per step run H, H sources, interior E, PML split E, edge E and Ez sources, then the probes.
    void SerialBackend::advance(std::size_t nsteps, std::span<const PointInjection> points,
                                std::span<const FloatType> values)
    {
        if (!m_has_coeffs) {
            throw std::logic_error("SerialBackend: step before the coefficients were uploaded");
        }
        const CoeffsView coeffs(m_coeff_ez_x1.data(), m_coeff_ez_x2.data(), m_coeff_ez_y1.data(),
                                m_coeff_ez_y2.data(), m_coeff_hx_1.data(), m_coeff_hx_2.data(),
                                m_coeff_hy_1.data(), m_coeff_hy_2.data());
        const std::size_t pml = m_boundary == Boundary::Pml ? m_pml_thickness : 0;
        m_records.reserve(m_records.size() + nsteps * m_probes.size());
        for (std::size_t n = 0; n < nsteps; ++n) {
            const FloatType *row = values.data() + n * points.size();
            serial_update_h(m_fields, coeffs);
            add_sources(m_fields, points, row, true);
            serial_update_e(m_fields, coeffs, pml);
            serial_update_pml(m_fields, coeffs, pml);
            serial_update_boundary(m_fields, coeffs, m_boundary);
            add_sources(m_fields, points, row, false);
            for (const PointProbe &probe : m_probes) {
                m_records.push_back(read_point(m_fields, probe));
            }
        }
    }
}
