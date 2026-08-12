/*
 * C++ file for the 2D TMz stepper, one advance of the fields part by part.
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
#include "fdtd/TMz/backends/fdtd_tmz_stepper2d.h"

// Standard library for sizes and buffers.
#include <cstddef>
#include <vector>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Helpers private to this file.
    namespace
    {
        // Sort point indices by the part of their cell, stable, as offsets and an order.
        template <typename Point>
        void bucket(const Partition2D &partition, const Grid2D &grid, std::span<const Point> points,
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
        std::span<FloatType> field_of(Fields2D &fields, Component component) noexcept
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

    // Keep the advance and sort its sources and probes by part.
    Stepper2D::Stepper2D(Fields2D &fields, CoeffsView coeffs, std::size_t pml, Boundary boundary,
                         const Partition2D &partition, std::span<const PointInjection> sources,
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
    void Stepper2D::add_sources(std::size_t part, std::size_t n, bool magnetic) const noexcept
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
    void Stepper2D::record_probes(std::size_t part, std::size_t n) const noexcept
    {
        FloatType *row = m_records + n * m_probes.size();
        for (std::size_t s = m_probe_offsets[part]; s < m_probe_offsets[part + 1]; ++s) {
            const std::size_t k = m_probe_order[s];
            const PointProbe &probe = m_probes[k];
            row[k] = field_of(m_fields, probe.component)[probe.index];
        }
    }
}
