/*
 * C++ header for the partitions of a 2D TMz grid into parts of blocks.
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

#ifndef FDTD_TMZ_BACKEND_CPU_PARTITION_H
#define FDTD_TMZ_BACKEND_CPU_PARTITION_H

#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_block.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"

#include <cstddef>
#include <span>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /** Namespace for what the 2D TMz CPU backends share. */
            namespace cpu
            {
                /**
                 * A cut of a grid into parts, each part one to three blocks, every cell in one
                 * part.
                 *
                 * A part is the unit of work a backend hands to a thread or a task. Tiles are a
                 * regular cut, rows being tiles of nx by 1, and know their neighbours; ranges are
                 * contiguous runs of cells in row-major order, a partial row, whole rows and a
                 * partial row.
                 */
                class Partition
                {
                public:
                    /**
                     * Cut a grid into one part, the whole grid.
                     *
                     * @param grid Grid to cut.
                     * @return The partition.
                     */
                    static Partition whole(const Grid &grid);

                    /**
                     * Cut a grid into tiles of tileX by tileY cells, row-major, the last ones
                     * clipped.
                     *
                     * @param grid Grid to cut.
                     * @param tileX Tile width in cells, positive.
                     * @param tileY Tile height in cells, positive.
                     * @return The partition.
                     * @throws std::invalid_argument When a tile side is zero.
                     */
                    static Partition tiles(const Grid &grid, std::size_t tileX, std::size_t tileY);

                    /**
                     * Cut a grid into parts contiguous ranges of cells in row-major order, part p
                     * holding cells p N / parts up to (p + 1) N / parts of the N = nx ny. A range
                     * may be empty when there are more parts than cells.
                     *
                     * @param grid Grid to cut.
                     * @param parts Number of parts, positive.
                     * @return The partition.
                     * @throws std::invalid_argument When parts is zero.
                     */
                    static Partition ranges(const Grid &grid, std::size_t parts);

                    /**
                     * Number of parts.
                     *
                     * @return The part count.
                     */
                    std::size_t size() const noexcept;

                    /**
                     * Blocks of one part.
                     *
                     * @param part Part below size().
                     * @return The part's blocks, none for an empty range.
                     */
                    std::span<const Block> blocks(std::size_t part) const noexcept;

                    /**
                     * Part that holds a cell.
                     *
                     * @param i Column, below nx.
                     * @param j Row, below ny.
                     * @return The part.
                     */
                    std::size_t owner(std::size_t i, std::size_t j) const noexcept;

                    /**
                     * Tiles along x, part p sitting at column p % tiles_x() of the tiles.
                     *
                     * @return The tile column count, zero for ranges.
                     */
                    std::size_t tiles_x() const noexcept;

                    /**
                     * Tiles along y.
                     *
                     * @return The tile row count, zero for ranges.
                     */
                    std::size_t tiles_y() const noexcept;

                private:
                    /**
                     * Create an empty partition of a grid.
                     *
                     * @param grid Grid to cut.
                     */
                    explicit Partition(const Grid &grid);

                    /**
                     * Close the part made of the blocks added since the last one.
                     */
                    void close_part();

                    /** Cells along x of the grid. */
                    std::size_t m_nx;

                    /** Tile width, zero for ranges. */
                    std::size_t m_tile_x = 0;

                    /** Tile height, zero for ranges. */
                    std::size_t m_tile_y = 0;

                    /** Tile columns, zero for ranges. */
                    std::size_t m_tiles_x = 0;

                    /** Tile rows, zero for ranges. */
                    std::size_t m_tiles_y = 0;

                    /** First row-major cell of each range, empty for tiles. */
                    std::vector<std::size_t> m_starts;

                    /** Blocks of every part, part by part. */
                    std::vector<Block> m_blocks;

                    /** Part p's blocks run from m_offsets[p] to m_offsets[p + 1]. */
                    std::vector<std::size_t> m_offsets;
                };
            }
        }
    }
}

#endif
