/*
 * C++ file for the 2D TMz serial backend.
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
#include "fdtd/TMz/serial/fdtd_tmz_serial_backend.h"

// Serial update and boundary routines.
#include "fdtd/TMz/serial/fdtd_tmz_serial_boundary.h"
#include "fdtd/TMz/serial/fdtd_tmz_serial_update.h"

// Standard library for copies and errors.
#include <algorithm>
#include <span>
#include <stdexcept>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
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

    // Allocate the backend's fields and coefficient arrays.
    SerialBackend::SerialBackend(const Grid2D &grid)
        : m_fields(grid)
        , m_ce1(grid.size())
        , m_ce2(grid.size())
        , m_ch1(grid.size())
        , m_ch2(grid.size())
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
    }

    // Copy the fields out.
    void SerialBackend::download(Fields2D &fields) const
    {
        require_same_grid(grid(), fields.grid());
        copy_field(m_fields.ez(), fields.ez());
        copy_field(m_fields.hx(), fields.hx());
        copy_field(m_fields.hy(), fields.hy());
    }

    // Copy the coefficient arrays in.
    void SerialBackend::upload(const Coeffs2D &coeffs)
    {
        require_same_grid(grid(), coeffs.grid());
        copy_field(coeffs.ce1(), {m_ce1.data(), m_ce1.size()});
        copy_field(coeffs.ce2(), {m_ce2.data(), m_ce2.size()});
        copy_field(coeffs.ch1(), {m_ch1.data(), m_ch1.size()});
        copy_field(coeffs.ch2(), {m_ch2.data(), m_ch2.size()});
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

    // Run H, the H sources, the interior E, the edge E and the Ez sources once per step.
    void SerialBackend::advance(std::size_t nsteps, std::span<const PointInjection> points,
                                std::span<const FloatType> values)
    {
        if (!m_has_coeffs) {
            throw std::logic_error("SerialBackend: step before the coefficients were uploaded");
        }
        const CoeffsView coeffs(m_ce1.data(), m_ce2.data(), m_ch1.data(), m_ch2.data());
        for (std::size_t n = 0; n < nsteps; ++n) {
            const FloatType *row = values.data() + n * points.size();
            serial_update_h(m_fields, coeffs);
            add_sources(m_fields, points, row, true);
            serial_update_e(m_fields, coeffs);
            serial_update_boundary(m_fields, coeffs, m_boundary);
            add_sources(m_fields, points, row, false);
        }
    }
}
