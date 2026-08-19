/*
 * C++ file for the 2D TMz serial backend: the stencil kernels, the partition and the stepper.
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

// Class, struct and kernel declarations.
#include "fdtd/TMz/backends/fdtd_tmz_backend_serial.h"

// Standard library for copies, searches, sizes, errors, string building and buffers.
#include <algorithm>
#include <cstddef>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

// Namespace for the serial FDTD algorithm.
namespace fdtd::TMz::backends::serial
{
    // Helpers private to this file.
    namespace
    {
        // Visit each edge cell of a block once: the first and last rows, then the row ends between.
        template <typename Visit>
        void for_each_edge(const Grid &grid, const Block &block, Visit visit)
        {
            const std::size_t nx = grid.nx();
            const std::size_t ny = grid.ny();
            if (block.empty()) {
                return;
            }
            for (std::size_t i = block.i0; i < block.i1; ++i) {
                if (block.j0 == 0) {
                    visit(i, std::size_t(0));
                }
                if (ny > 1 && block.j1 == ny) {
                    visit(i, ny - 1);
                }
            }
            const std::size_t jBegin = std::max<std::size_t>(block.j0, 1);
            const std::size_t jEnd = std::min(block.j1, ny - 1);
            for (std::size_t j = jBegin; j < jEnd; ++j) {
                if (block.i0 == 0) {
                    visit(std::size_t(0), j);
                }
                if (nx > 1 && block.i1 == nx) {
                    visit(nx - 1, j);
                }
            }
        }

        // Sort point indices by the part of their cell, stable, as offsets and an order.
        template <typename Point>
        void bucket(const Partition &partition, const Grid &grid, std::span<const Point> points,
                    std::vector<std::size_t> &offsets, std::vector<std::size_t> &order)
        {
            const std::size_t pitch = grid.pitch();
            std::vector<std::size_t> owners(points.size());
            offsets.assign(partition.size() + 1, 0);
            for (std::size_t k = 0; k < points.size(); ++k) {
                const std::size_t index = points[k].index;
                owners[k] = partition.owner(index % pitch, index / pitch);
                ++offsets[owners[k] + 1];
            }
            for (std::size_t p = 0; p < partition.size(); ++p) {
                offsets[p + 1] += offsets[p];
            }
            std::vector<std::size_t> next(offsets.begin(), offsets.end() - 1);
            order.assign(points.size(), 0);
            for (std::size_t k = 0; k < points.size(); ++k) {
                order[next[owners[k]]++] = k;
            }
        }

        // Return the field of a component.
        std::span<FloatType> field_of(Fields &fields, Component component) noexcept
        {
            switch (component) {
                case Component::Hx:
                    return fields.hx();
                case Component::Hy:
                    return fields.hy();
                case Component::Ez:
                    break;
            }
            return fields.ez();
        }
    }

    // Take the arrays of a coefficient set.
    CoeffsView::CoeffsView(const Coeffs &coeffs) noexcept
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

    // Zero the block's edge Ez and its parts, or update it reading H outside the domain as zero.
    void update_boundary_block(Fields &fields, CoeffsView coeffs, const Block &block,
                               Boundary boundary) noexcept
    {
        const Grid &grid = fields.grid();
        FloatType *__restrict Ez = fields.ez().data();

        if (boundary != Boundary::Pmc) {
            FloatType *__restrict Ezx = fields.ezx().data();
            FloatType *__restrict Ezy = fields.ezy().data();
            for_each_edge(grid, block, [&](std::size_t i, std::size_t j) {
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

        for_each_edge(grid, block, [&](std::size_t i, std::size_t j) {
            const std::size_t k = grid.index(i, j);
            const FloatType hyRight = i + 1 < nx ? Hy[k] : FloatType(0);
            const FloatType hyLeft = i > 0 ? Hy[k - 1] : FloatType(0);
            const FloatType hxTop = j + 1 < ny ? Hx[k] : FloatType(0);
            const FloatType hxBottom = j > 0 ? Hx[k - pitch] : FloatType(0);
            const FloatType curl = (hyRight - hyLeft) * invDx - (hxTop - hxBottom) * invDy;
            Ez[k] = Cex1[k] * Ez[k] + Cex2[k] * curl;
        });
    }

    // Start with no parts.
    Partition::Partition(const Grid &grid)
        : m_nx(grid.nx())
        , m_offsets{0}
    {
        // Empty.
    }

    // Cut into one tile as large as the grid.
    Partition Partition::whole(const Grid &grid)
    {
        return tiles(grid, grid.nx(), grid.ny());
    }

    // Cut into tiles row by row of tiles, clipping the last column and row.
    Partition Partition::tiles(const Grid &grid, std::size_t tileX, std::size_t tileY)
    {
        if (tileX == 0 || tileY == 0) {
            throw std::invalid_argument("Partition: tile sides must be positive");
        }
        Partition partition(grid);
        partition.m_tile_x = tileX;
        partition.m_tile_y = tileY;
        partition.m_tiles_x = (grid.nx() + tileX - 1) / tileX;
        partition.m_tiles_y = (grid.ny() + tileY - 1) / tileY;
        for (std::size_t j = 0; j < grid.ny(); j += tileY) {
            for (std::size_t i = 0; i < grid.nx(); i += tileX) {
                partition.m_blocks.push_back({
                    .i0 = i,
                    .i1 = std::min(i + tileX, grid.nx()),
                    .j0 = j,
                    .j1 = std::min(j + tileY, grid.ny()),
                });
                partition.close_part();
            }
        }
        return partition;
    }

    // Cut the row-major cells into even ranges, each a partial row, whole rows and a partial row.
    Partition Partition::ranges(const Grid &grid, std::size_t parts)
    {
        if (parts == 0) {
            throw std::invalid_argument("Partition: the part count must be positive");
        }
        const std::size_t nx = grid.nx();
        const std::size_t cells = nx * grid.ny();
        Partition partition(grid);
        for (std::size_t p = 0; p < parts; ++p) {
            const std::size_t begin = p * cells / parts;
            const std::size_t end = (p + 1) * cells / parts;
            partition.m_starts.push_back(begin);
            std::size_t row = begin / nx;
            const std::size_t col = begin % nx;
            const std::size_t lastRow = end / nx;
            const std::size_t lastCol = end % nx;
            if (begin == end) {
                partition.close_part();
                continue;
            }
            if (row == lastRow) {
                partition.m_blocks.push_back({.i0 = col, .i1 = lastCol, .j0 = row, .j1 = row + 1});
                partition.close_part();
                continue;
            }
            if (col > 0) {
                partition.m_blocks.push_back({.i0 = col, .i1 = nx, .j0 = row, .j1 = row + 1});
                ++row;
            }
            if (lastRow > row) {
                partition.m_blocks.push_back({.i0 = 0, .i1 = nx, .j0 = row, .j1 = lastRow});
            }
            if (lastCol > 0) {
                partition.m_blocks.push_back(
                    {.i0 = 0, .i1 = lastCol, .j0 = lastRow, .j1 = lastRow + 1});
            }
            partition.close_part();
        }
        return partition;
    }

    // Mark the end of the current part's blocks.
    void Partition::close_part()
    {
        m_offsets.push_back(m_blocks.size());
    }

    // Return the number of parts.
    std::size_t Partition::size() const noexcept
    {
        return m_offsets.size() - 1;
    }

    // Return the blocks between the part's offsets.
    std::span<const Block> Partition::blocks(std::size_t part) const noexcept
    {
        const std::size_t first = m_offsets[part];
        return std::span(m_blocks).subspan(first, m_offsets[part + 1] - first);
    }

    // Find the tile of the cell, or the last range that starts at or before it.
    std::size_t Partition::owner(std::size_t i, std::size_t j) const noexcept
    {
        if (m_tile_x > 0) {
            return (j / m_tile_y) * m_tiles_x + i / m_tile_x;
        }
        const std::size_t cell = j * m_nx + i;
        const auto it = std::upper_bound(m_starts.begin(), m_starts.end(), cell);
        return static_cast<std::size_t>(it - m_starts.begin()) - 1;
    }

    // Return the tile column count.
    std::size_t Partition::tiles_x() const noexcept
    {
        return m_tiles_x;
    }

    // Return the tile row count.
    std::size_t Partition::tiles_y() const noexcept
    {
        return m_tiles_y;
    }

    // Keep the advance and sort its sources and probes by part.
    Stepper::Stepper(Fields &fields, CoeffsView coeffs, std::size_t pml, Boundary boundary,
                     const Partition &partition, std::span<const PointInjection> sources,
                     std::span<const FloatType> values, std::span<const PointProbe> probes,
                     FloatType *records)
        : m_fields(fields)
        , m_coeffs(coeffs)
        , m_pml(pml)
        , m_boundary(boundary)
        , m_partition(partition)
        , m_sources(sources)
        , m_values(values)
        , m_probes(probes)
        , m_records(records)
    {
        bucket(partition, fields.grid(), sources, m_source_offsets, m_source_order);
        bucket(partition, fields.grid(), probes, m_probe_offsets, m_probe_order);
    }

    // Add the part's values of step n to its H or Ez sources.
    void Stepper::add_sources(std::size_t part, std::size_t n, bool magnetic) const noexcept
    {
        const FloatType *row = m_values.data() + n * m_sources.size();
        for (std::size_t s = m_source_offsets[part]; s < m_source_offsets[part + 1]; ++s) {
            const std::size_t k = m_source_order[s];
            const PointInjection &point = m_sources[k];
            if ((point.component != Component::Ez) == magnetic) {
                field_of(m_fields, point.component)[point.index] += row[k];
            }
        }
    }

    // Copy the part's probe values of step n into the records.
    void Stepper::record_probes(std::size_t part, std::size_t n) const noexcept
    {
        FloatType *row = m_records + n * m_probes.size();
        for (std::size_t s = m_probe_offsets[part]; s < m_probe_offsets[part + 1]; ++s) {
            const std::size_t k = m_probe_order[s];
            const PointProbe &probe = m_probes[k];
            row[k] = field_of(m_fields, probe.component)[probe.index];
        }
    }
}

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
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
    SerialBackend::SerialBackend(const Grid &grid)
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
    const Grid &SerialBackend::grid() const noexcept
    {
        return m_fields.grid();
    }

    // Copy the fields in.
    void SerialBackend::upload(const Fields &fields)
    {
        require_same_grid(name(), grid(), fields.grid());
        copy_field(fields.ez(), m_fields.ez());
        copy_field(fields.hx(), m_fields.hx());
        copy_field(fields.hy(), m_fields.hy());
        copy_field(fields.ezx(), m_fields.ezx());
        copy_field(fields.ezy(), m_fields.ezy());
    }

    // Copy the fields out.
    void SerialBackend::download(Fields &fields) const
    {
        require_same_grid(name(), grid(), fields.grid());
        copy_field(m_fields.ez(), fields.ez());
        copy_field(m_fields.hx(), fields.hx());
        copy_field(m_fields.hy(), fields.hy());
        copy_field(m_fields.ezx(), fields.ezx());
        copy_field(m_fields.ezy(), fields.ezy());
    }

    // Copy the coefficient arrays in.
    void SerialBackend::upload(const Coeffs &coeffs)
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
        const serial::Partition partition = serial::Partition::whole(grid());
        const serial::Stepper stepper(fields(), coeffs(), active_pml(), boundary(), partition,
                                      points, values, probes(), records);
        for (std::size_t n = 0; n < nsteps; ++n) {
            stepper.h_phase(0, n);
            stepper.e_phase(0, n);
        }
    }

    // Model the block kernels, which make an H pass and an E pass.
    StepTraffic SerialBackend::step_traffic() const noexcept
    {
        return two_pass_traffic(grid(), active_pml(), m_boundary);
    }

    // Keep the probe cells and clear the records.
    void SerialBackend::reset_probes(std::span<const PointProbe> points)
    {
        m_probes.assign(points.begin(), points.end());
        m_records.clear();
    }

    // Throw unless coefficients were uploaded.
    void SerialBackend::require_coeffs() const
    {
        if (!m_has_coeffs) {
            std::ostringstream message;
            message << name();
            message << " backend: step before the coefficients were uploaded";
            throw std::logic_error(message.str());
        }
    }

    // Return the fields.
    Fields &SerialBackend::fields() noexcept
    {
        return m_fields;
    }

    // View the coefficient arrays.
    serial::CoeffsView SerialBackend::coeffs() const noexcept
    {
        return {m_coeff_ez_x1.data(), m_coeff_ez_x2.data(), m_coeff_ez_y1.data(),
                m_coeff_ez_y2.data(), m_coeff_hx_1.data(),  m_coeff_hx_2.data(),
                m_coeff_hy_1.data(),  m_coeff_hy_2.data()};
    }

    // Return the PML thickness under Boundary::Pml, zero otherwise.
    std::size_t SerialBackend::active_pml() const noexcept
    {
        return m_boundary == Boundary::Pml ? m_pml_thickness : 0;
    }

    // Return the probe cells.
    std::span<const PointProbe> SerialBackend::probes() const noexcept
    {
        return m_probes;
    }

    // Grow the records by nsteps steps of zeros and point at the first new one.
    FloatType *SerialBackend::add_records(std::size_t nsteps)
    {
        const std::size_t first = m_records.size();
        m_records.resize(first + nsteps * m_probes.size());
        return m_records.data() + first;
    }
}
