/*
 * C++ file for the 2D TMz CUDA backend's level v3, table.
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
#include "fdtd/TMz/backends/fdtd_tmz_backend_cuda.h"

// Standard library for bounds, sizes and the optional tuple.
#include <algorithm>
#include <cstddef>
#include <optional>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Helpers private to this file.
    namespace
    {
        // Copy of a table entry, or none for a null one.
        std::optional<gpu::CoeffTuple> tuple_of(const gpu::CoeffTuple *entry)
        {
            if (entry == nullptr) {
                return std::nullopt;
            }
            return *entry;
        }
    }

    // Build the table, copy its entries and the byte per cell, and keep the interior's tuples.
    void CudaBackend::upload_table(const Coeffs &coeffs)
    {
        const gpu::CoeffTable table(coeffs);
        gpu::reserve(m_table_entries, std::max<std::size_t>(table.entries().size(), 1));
        gpu::upload(m_table_entries, table.entries(), m_stream);
        gpu::reserve(m_table_index, m_pitch * m_grid.ny());
        gpu::upload_rows(m_table_index, m_pitch, table.index(), m_grid.pitch(), m_grid.nx(),
                         m_grid.ny(), m_stream);
        m_uniform_pml =
            tuple_of(table.uniform_entry(std::max<std::size_t>(coeffs.pml_thickness(), 1)));
        m_uniform_plain = tuple_of(table.uniform_entry(1));
    }

    // Read the interior's one tuple from the cell's position when it shares one, else the byte.
    gpu::DeviceCoeffs CudaBackend::table_coeffs(gpu::DeviceCoeffs arrays) const noexcept
    {
        const std::optional<gpu::CoeffTuple> &uniform = uniform_tuple();
        if (uniform.has_value()) {
            arrays.source = gpu::CoeffSource::Uniform;
            arrays.uniform = *uniform;
        } else {
            arrays.source = gpu::CoeffSource::Indexed;
            arrays.table = m_table_entries.data();
            arrays.index = m_table_index.data();
        }
        return arrays;
    }
}
