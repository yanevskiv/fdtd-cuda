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
        : CoeffsView(coeffs.coeff_ez_x1().data(), coeffs.coeff_ez_x2().data(),
                     coeffs.coeff_ez_y1().data(), coeffs.coeff_ez_y2().data(),
                     coeffs.coeff_hx_1().data(), coeffs.coeff_hx_2().data(),
                     coeffs.coeff_hy_1().data(), coeffs.coeff_hy_2().data())
    {
        // Empty.
    }

    // Take eight arrays.
    CoeffsView::CoeffsView(const FloatType *ezX1, const FloatType *ezX2, const FloatType *ezY1,
                           const FloatType *ezY2, const FloatType *hx1, const FloatType *hx2,
                           const FloatType *hy1, const FloatType *hy2) noexcept
        : coeff_ez_x1(ezX1)
        , coeff_ez_x2(ezX2)
        , coeff_ez_y1(ezY1)
        , coeff_ez_y2(ezY2)
        , coeff_hx_1(hx1)
        , coeff_hx_2(hx2)
        , coeff_hy_1(hy1)
        , coeff_hy_2(hy2)
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
        const FloatType *__restrict Chx1 = coeffs.coeff_hx_1;
        const FloatType *__restrict Chx2 = coeffs.coeff_hx_2;
        const FloatType *__restrict Chy1 = coeffs.coeff_hy_1;
        const FloatType *__restrict Chy2 = coeffs.coeff_hy_2;

        for (std::size_t j = 0; j + 1 < ny; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t k = row + i;
                Hx[k] = Chx1[k] * Hx[k] - Chx2[k] * invDy * (Ez[k + pitch] - Ez[k]);
            }
        }

        for (std::size_t j = 0; j < ny; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = 0; i + 1 < nx; ++i) {
                const std::size_t k = row + i;
                Hy[k] = Chy1[k] * Hy[k] + Chy2[k] * invDx * (Ez[k + 1] - Ez[k]);
            }
        }
    }

    // Update the interior Ez row by row, inside the PML when there is one.
    void serial_update_e(Fields2D &fields, CoeffsView coeffs, std::size_t pml) noexcept
    {
        const Grid2D &grid = fields.grid();
        const std::size_t pitch = grid.pitch();
        const std::size_t lo = std::max<std::size_t>(pml, 1);
        const std::size_t iEnd = grid.nx() - std::min(lo, grid.nx());
        const std::size_t jEnd = grid.ny() - std::min(lo, grid.ny());
        const FloatType invDx = FloatType(1) / grid.dx();
        const FloatType invDy = FloatType(1) / grid.dy();

        FloatType *__restrict Ez = fields.ez().data();
        const FloatType *__restrict Hx = fields.hx().data();
        const FloatType *__restrict Hy = fields.hy().data();
        const FloatType *__restrict Cex1 = coeffs.coeff_ez_x1;
        const FloatType *__restrict Cex2 = coeffs.coeff_ez_x2;

        for (std::size_t j = lo; j < jEnd; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = lo; i < iEnd; ++i) {
                const std::size_t k = row + i;
                const FloatType curl =
                    (Hy[k] - Hy[k - 1]) * invDx - (Hx[k] - Hx[k - pitch]) * invDy;
                Ez[k] = Cex1[k] * Ez[k] + Cex2[k] * curl;
            }
        }
    }

    // Update Ezx and Ezy over a rectangle of non-edge cells row by row, and Ez as their sum.
    static void split_update_rect(Fields2D &fields, CoeffsView coeffs, std::size_t i0,
                                  std::size_t i1, std::size_t j0, std::size_t j1) noexcept
    {
        const Grid2D &grid = fields.grid();
        const std::size_t pitch = grid.pitch();
        const FloatType invDx = FloatType(1) / grid.dx();
        const FloatType invDy = FloatType(1) / grid.dy();

        FloatType *__restrict Ez = fields.ez().data();
        FloatType *__restrict Ezx = fields.ezx().data();
        FloatType *__restrict Ezy = fields.ezy().data();
        const FloatType *__restrict Hx = fields.hx().data();
        const FloatType *__restrict Hy = fields.hy().data();
        const FloatType *__restrict Cex1 = coeffs.coeff_ez_x1;
        const FloatType *__restrict Cex2 = coeffs.coeff_ez_x2;
        const FloatType *__restrict Cey1 = coeffs.coeff_ez_y1;
        const FloatType *__restrict Cey2 = coeffs.coeff_ez_y2;

        for (std::size_t j = j0; j < j1; ++j) {
            const std::size_t row = j * pitch;
            for (std::size_t i = i0; i < i1; ++i) {
                const std::size_t k = row + i;
                Ezx[k] = Cex1[k] * Ezx[k] + Cex2[k] * invDx * (Hy[k] - Hy[k - 1]);
                Ezy[k] = Cey1[k] * Ezy[k] - Cey2[k] * invDy * (Hx[k] - Hx[k - pitch]);
                Ez[k] = Ezx[k] + Ezy[k];
            }
        }
    }

    // Run the split update over the bottom and top slabs across the width, then over the
    // left and right slabs between them; a layer under two cells has no non-edge slab.
    void serial_update_pml(Fields2D &fields, CoeffsView coeffs, std::size_t pml) noexcept
    {
        if (pml < 2) {
            return;
        }
        const std::size_t nx = fields.grid().nx();
        const std::size_t ny = fields.grid().ny();
        split_update_rect(fields, coeffs, 1, nx - 1, 1, pml);
        split_update_rect(fields, coeffs, 1, nx - 1, ny - pml, ny - 1);
        split_update_rect(fields, coeffs, 1, pml, pml, ny - pml);
        split_update_rect(fields, coeffs, nx - pml, nx - 1, pml, ny - pml);
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

    // Zero the edge Ez and its split parts under PEC, or apply the Ez update with H outside
    // the domain read as zero.
    void serial_update_boundary(Fields2D &fields, CoeffsView coeffs, Boundary boundary) noexcept
    {
        const Grid2D &grid = fields.grid();
        FloatType *__restrict Ez = fields.ez().data();

        if (boundary != Boundary::Pmc) {
            FloatType *__restrict Ezx = fields.ezx().data();
            FloatType *__restrict Ezy = fields.ezy().data();
            for_each_edge(grid, [&](std::size_t i, std::size_t j) {
                const std::size_t k = grid.index(i, j);
                Ez[k] = 0;
                Ezx[k] = 0;
                Ezy[k] = 0;
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
        const FloatType *__restrict Cex1 = coeffs.coeff_ez_x1;
        const FloatType *__restrict Cex2 = coeffs.coeff_ez_x2;

        for_each_edge(grid, [&](std::size_t i, std::size_t j) {
            const std::size_t k = grid.index(i, j);
            const FloatType hyRight = i + 1 < nx ? Hy[k] : FloatType(0);
            const FloatType hyLeft = i > 0 ? Hy[k - 1] : FloatType(0);
            const FloatType hxTop = j + 1 < ny ? Hx[k] : FloatType(0);
            const FloatType hxBottom = j > 0 ? Hx[k - pitch] : FloatType(0);
            const FloatType curl = (hyRight - hyLeft) * invDx - (hxTop - hxBottom) * invDy;
            Ez[k] = Cex1[k] * Ez[k] + Cex2[k] * curl;
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

    // Run H, the H sources, the interior E, the split E of an active PML, the edge E and the
    // Ez sources once per step, then record the probes.
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
