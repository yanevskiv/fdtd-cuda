/*
 * C++ file for the partitions of a 2D TMz grid into parts of blocks.
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
#include "fdtd/TMz/backends/fdtd_tmz_partition2d.h"

// Standard library for searches, sizes and errors.
#include <algorithm>
#include <cstddef>
#include <stdexcept>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Start with no parts.
    Partition2D::Partition2D(const Grid2D &grid)
        : m_nx(grid.nx())
        , m_offsets{0}
    {
        // Empty.
    }

    // Cut into one tile as large as the grid.
    Partition2D Partition2D::whole(const Grid2D &grid)
    {
        return tiles(grid, grid.nx(), grid.ny());
    }

    // Cut into tiles row by row of tiles, clipping the last column and row.
    Partition2D Partition2D::tiles(const Grid2D &grid, std::size_t tileX, std::size_t tileY)
    {
        if (tileX == 0 || tileY == 0) {
            throw std::invalid_argument("Partition2D: tile sides must be positive");
        }
        Partition2D partition(grid);
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
    Partition2D Partition2D::ranges(const Grid2D &grid, std::size_t parts)
    {
        if (parts == 0) {
            throw std::invalid_argument("Partition2D: the part count must be positive");
        }
        const std::size_t nx = grid.nx();
        const std::size_t cells = nx * grid.ny();
        Partition2D partition(grid);
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
    void Partition2D::close_part()
    {
        m_offsets.push_back(m_blocks.size());
    }

    // Return the number of parts.
    std::size_t Partition2D::size() const noexcept
    {
        return m_offsets.size() - 1;
    }

    // Return the blocks between the part's offsets.
    std::span<const Block2D> Partition2D::blocks(std::size_t part) const noexcept
    {
        const std::size_t first = m_offsets[part];
        return std::span(m_blocks).subspan(first, m_offsets[part + 1] - first);
    }

    // Find the tile of the cell, or the last range that starts at or before it.
    std::size_t Partition2D::owner(std::size_t i, std::size_t j) const noexcept
    {
        if (m_tile_x > 0) {
            return (j / m_tile_y) * m_tiles_x + i / m_tile_x;
        }
        const std::size_t cell = j * m_nx + i;
        const auto it = std::upper_bound(m_starts.begin(), m_starts.end(), cell);
        return static_cast<std::size_t>(it - m_starts.begin()) - 1;
    }

    // Return the tile column count.
    std::size_t Partition2D::tiles_x() const noexcept
    {
        return m_tiles_x;
    }

    // Return the tile row count.
    std::size_t Partition2D::tiles_y() const noexcept
    {
        return m_tiles_y;
    }
}
