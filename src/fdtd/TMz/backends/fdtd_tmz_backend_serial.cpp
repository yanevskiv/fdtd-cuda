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
    // Take the arrays of a coefficient set.
    CoeffsView::CoeffsView(const Coeffs2D &coeffs) noexcept
        : CoeffsView(coeffs.ce1().data(), coeffs.ce2().data(), coeffs.ch1().data(),
                     coeffs.ch2().data())
    {
        // Empty.
    }

    // Take four arrays.
    CoeffsView::CoeffsView(const FloatType *ce1Data, const FloatType *ce2Data,
                           const FloatType *ch1Data, const FloatType *ch2Data) noexcept
        : ce1(ce1Data)
        , ce2(ce2Data)
        , ch1(ch1Data)
        , ch2(ch2Data)
    {
        // Empty.
    }

    // Update Hx row by row below the last row, then Hy row by row left of the last column.
    void serial_update_h(Fields2D &fields, CoeffsView coeffs) noexcept
    {
        const Grid2D &grid = fields.grid();
        const std::size_t nx = grid.nx();
        const std::size_t ny = grid.ny();
        const std::size_t pitch = grid.pitch();
        const FloatType invDx = FloatType(1) / grid.dx();
        const FloatType invDy = FloatType(1) / grid.dy();

        const FloatType *__restrict Ez = fields.ez().data();
        FloatType *__restrict Hx = fields.hx().data();
        FloatType *__restrict Hy = fields.hy().data();
        const FloatType *__restrict Ch1 = coeffs.ch1;
        const FloatType *__restrict Ch2 = coeffs.ch2;

        for (std::size_t j = 0; j + 1 < ny; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t k = row + i;
                Hx[k] = Ch1[k] * Hx[k] - Ch2[k] * invDy * (Ez[k + pitch] - Ez[k]);
            }
        }

        for (std::size_t j = 0; j < ny; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = 0; i + 1 < nx; ++i) {
                const std::size_t k = row + i;
                Hy[k] = Ch1[k] * Hy[k] + Ch2[k] * invDx * (Ez[k + 1] - Ez[k]);
            }
        }
    }

    // Update the interior Ez row by row.
    void serial_update_e(Fields2D &fields, CoeffsView coeffs) noexcept
    {
        const Grid2D &grid = fields.grid();
        const std::size_t nx = grid.nx();
        const std::size_t ny = grid.ny();
        const std::size_t pitch = grid.pitch();
        const FloatType invDx = FloatType(1) / grid.dx();
        const FloatType invDy = FloatType(1) / grid.dy();

        FloatType *__restrict Ez = fields.ez().data();
        const FloatType *__restrict Hx = fields.hx().data();
        const FloatType *__restrict Hy = fields.hy().data();
        const FloatType *__restrict Ce1 = coeffs.ce1;
        const FloatType *__restrict Ce2 = coeffs.ce2;

        for (std::size_t j = 1; j + 1 < ny; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = 1; i + 1 < nx; ++i) {
                const std::size_t k = row + i;
                const FloatType curl =
                    (Hy[k] - Hy[k - 1]) * invDx - (Hx[k] - Hx[k - pitch]) * invDy;
                Ez[k] = Ce1[k] * Ez[k] + Ce2[k] * curl;
            }
        }
    }

    // Visit the edge cells once each: the first and last rows whole, then the first and
    // last cells of the rows between, which also covers grids one cell wide.
    template <typename Visit> static void for_each_edge(const Grid2D &grid, Visit visit)
    {
        const std::size_t nx = grid.nx();
        const std::size_t ny = grid.ny();
        for (std::size_t i = 0; i < nx; ++i) {
            visit(i, std::size_t(0));
            if (ny > 1) {
                visit(i, ny - 1);
            }
        }
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            visit(std::size_t(0), j);
            if (nx > 1) {
                visit(nx - 1, j);
            }
        }
    }

    // Zero the edge under PEC, or apply the Ez update with H outside the domain read as zero.
    void serial_update_boundary(Fields2D &fields, CoeffsView coeffs, Boundary boundary) noexcept
    {
        const Grid2D &grid = fields.grid();
        FloatType *__restrict Ez = fields.ez().data();

        if (boundary != Boundary::Pmc) {
            for_each_edge(grid, [&](std::size_t i, std::size_t j) {
                Ez[grid.index(i, j)] = 0;
            });
            return;
        }

        const std::size_t nx = grid.nx();
        const std::size_t ny = grid.ny();
        const std::size_t pitch = grid.pitch();
        const FloatType invDx = FloatType(1) / grid.dx();
        const FloatType invDy = FloatType(1) / grid.dy();
        const FloatType *__restrict Hx = fields.hx().data();
        const FloatType *__restrict Hy = fields.hy().data();
        const FloatType *__restrict Ce1 = coeffs.ce1;
        const FloatType *__restrict Ce2 = coeffs.ce2;

        for_each_edge(grid, [&](std::size_t i, std::size_t j) {
            const std::size_t k = grid.index(i, j);
            const FloatType hyRight = i + 1 < nx ? Hy[k] : FloatType(0);
            const FloatType hyLeft = i > 0 ? Hy[k - 1] : FloatType(0);
            const FloatType hxTop = j + 1 < ny ? Hx[k] : FloatType(0);
            const FloatType hxBottom = j > 0 ? Hx[k - pitch] : FloatType(0);
            const FloatType curl = (hyRight - hyLeft) * invDx - (hxTop - hxBottom) * invDy;
            Ez[k] = Ce1[k] * Ez[k] + Ce2[k] * curl;
        });
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

    // Run H, the H sources, the interior E, the edge E and the Ez sources once per step,
    // then record the probes.
    void SerialBackend::advance(std::size_t nsteps, std::span<const PointInjection> points,
                                std::span<const FloatType> values)
    {
        if (!m_has_coeffs) {
            throw std::logic_error("SerialBackend: step before the coefficients were uploaded");
        }
        const CoeffsView coeffs(m_ce1.data(), m_ce2.data(), m_ch1.data(), m_ch2.data());
        m_records.reserve(m_records.size() + nsteps * m_probes.size());
        for (std::size_t n = 0; n < nsteps; ++n) {
            const FloatType *row = values.data() + n * points.size();
            serial_update_h(m_fields, coeffs);
            add_sources(m_fields, points, row, true);
            serial_update_e(m_fields, coeffs);
            serial_update_boundary(m_fields, coeffs, m_boundary);
            add_sources(m_fields, points, row, false);
            for (const PointProbe &probe : m_probes) {
                m_records.push_back(read_point(m_fields, probe));
            }
        }
    }
}
