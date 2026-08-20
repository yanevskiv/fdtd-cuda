/*
 * C++ file for the CUDA backend's table of distinct update coefficients, read in place of arrays.
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
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_table.h"

// Standard library for bit patterns, ordered maps, sizes, arrays and type choices.
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <type_traits>

// Namespace for what the 2D TMz CUDA backend's files share.
namespace fdtd::TMz::backends::gpu
{
    // Helpers private to this file.
    namespace
    {
        // Unsigned integer as wide as FloatType, for comparing values bit by bit.
        using Bits = std::conditional_t<sizeof(FloatType) == 8, std::uint64_t, std::uint32_t>;

        // Bit patterns of a tuple, so that equal keys hold bit-for-bit equal values.
        using Key = std::array<Bits, 6>;

        // Bit pattern of a value.
        Bits bits(FloatType value)
        {
            return std::bit_cast<Bits>(value);
        }
    }

    // Read each cell's tuple, give a new tuple the next entry while there is room.
    CoeffTable::CoeffTable(const Coeffs &coeffs)
        : m_grid(coeffs.grid())
        , m_index(coeffs.grid().size(), ARRAYS)
    {
        const auto ezX1 = coeffs.coeff_ez_x1();
        const auto ezX2 = coeffs.coeff_ez_x2();
        const auto ezY1 = coeffs.coeff_ez_y1();
        const auto ezY2 = coeffs.coeff_ez_y2();
        const auto hx1 = coeffs.coeff_hx_1();
        const auto hx2 = coeffs.coeff_hx_2();
        const auto hy1 = coeffs.coeff_hy_1();
        const auto hy2 = coeffs.coeff_hy_2();
        std::map<Key, std::uint8_t> found;
        for (std::size_t j = 0; j < m_grid.ny(); ++j) {
            for (std::size_t i = 0; i < m_grid.nx(); ++i) {
                const std::size_t k = m_grid.index(i, j);
                if (bits(ezX1[k]) != bits(ezY1[k]) || bits(ezX2[k]) != bits(ezY2[k])) {
                    continue;
                }
                const Key key{bits(ezX1[k]), bits(ezX2[k]), bits(hx1[k]), bits(hx2[k]), bits(hy1[k]), bits(hy2[k])};
                const auto it = found.find(key);
                if (it != found.end()) {
                    m_index[k] = it->second;
                } else if (m_entries.size() < CAPACITY) {
                    const auto entry = static_cast<std::uint8_t>(m_entries.size());
                    found.emplace(key, entry);
                    m_entries.push_back({ezX1[k], ezX2[k], hx1[k], hx2[k], hy1[k], hy2[k]});
                    m_index[k] = entry;
                }
            }
        }
    }

    // Return the grid.
    const Grid &CoeffTable::grid() const noexcept
    {
        return m_grid;
    }

    // Return the entries.
    std::span<const CoeffTuple> CoeffTable::entries() const noexcept
    {
        return m_entries;
    }

    // Return the index.
    std::span<const std::uint8_t> CoeffTable::index() const noexcept
    {
        return m_index;
    }

    // Check that every cell of the core takes the entry of its first cell.
    const CoeffTuple *CoeffTable::uniform_entry(std::size_t lo) const noexcept
    {
        if (m_grid.nx() <= 2 * lo + 1 || m_grid.ny() <= 2 * lo + 1) {
            return nullptr;
        }
        const std::uint8_t first = m_index[m_grid.index(lo, lo)];
        if (first == ARRAYS) {
            return nullptr;
        }
        for (std::size_t j = lo; j < m_grid.ny() - lo - 1; ++j) {
            for (std::size_t i = lo; i < m_grid.nx() - lo - 1; ++i) {
                if (m_index[m_grid.index(i, j)] != first) {
                    return nullptr;
                }
            }
        }
        return &m_entries[first];
    }
}
