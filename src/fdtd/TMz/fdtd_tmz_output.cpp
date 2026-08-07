/*
 * C++ file for the 2D TMz output files: probe CSV and binary snapshots.
 *
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

// Function declarations.
#include "fdtd/TMz/fdtd_tmz_output.h"

// Standard library for byte order, files, formatting and errors.
#include <bit>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    static_assert(std::endian::native == std::endian::little,
                  "snapshot files are written in the host's byte order, little-endian");

    // Helpers private to this file.
    namespace
    {
        // Append the bytes of a value to a buffer.
        template <typename T> void put(std::vector<char> &buffer, const T &value)
        {
            const auto *bytes = reinterpret_cast<const char *>(&value);
            buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
        }

        // Open a file for writing or throw.
        std::ofstream open_output(const std::filesystem::path &path, std::ios::openmode mode)
        {
            std::ofstream out(path, mode | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("cannot write " + path.string());
            }
            return out;
        }

        // Check that a stream is still good after writing.
        void require_written(const std::ofstream &out, const std::filesystem::path &path)
        {
            if (!out) {
                throw std::runtime_error("cannot write " + path.string());
            }
        }
    }

    // Write the header and Ez row by row, without the padding.
    void write_snapshot(const std::filesystem::path &path, const Fields2D &fields, std::size_t step,
                        double dt)
    {
        const Grid2D &grid = fields.grid();
        std::vector<char> header;
        header.reserve(SNAPSHOT_HEADER_SIZE);
        header.insert(header.end(), std::begin(SNAPSHOT_MAGIC), std::end(SNAPSHOT_MAGIC));
        put(header, SNAPSHOT_VERSION);
        put(header, static_cast<std::uint32_t>(sizeof(FloatType)));
        put(header, static_cast<std::uint64_t>(grid.nx()));
        put(header, static_cast<std::uint64_t>(grid.ny()));
        put(header, static_cast<std::uint64_t>(step));
        put(header, dt);
        put(header, static_cast<double>(grid.dx()));
        put(header, static_cast<double>(grid.dy()));

        std::ofstream out = open_output(path, std::ios::binary);
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        const auto ez = fields.ez();
        for (std::size_t j = 0; j < grid.ny(); ++j) {
            out.write(reinterpret_cast<const char *>(ez.data() + grid.index(0, j)),
                      static_cast<std::streamsize>(grid.nx() * sizeof(FloatType)));
        }
        require_written(out, path);
    }

    // Write the comment, the header and one row per value, with full precision.
    void write_probe_csv(const std::filesystem::path &path, std::string_view comment,
                         std::span<const double> times, std::span<const FloatType> values)
    {
        if (times.size() != values.size()) {
            throw std::invalid_argument("write_probe_csv: times and values differ in length");
        }
        std::ofstream out = open_output(path, std::ios::out);
        out.precision(std::numeric_limits<double>::max_digits10);
        if (!comment.empty()) {
            out << "# " << comment << '\n';
        }
        out << "step,time,ez\n";
        for (std::size_t n = 0; n < values.size(); ++n) {
            out << n + 1 << ',' << times[n] << ',' << static_cast<double>(values[n]) << '\n';
        }
        require_written(out, path);
    }
}
