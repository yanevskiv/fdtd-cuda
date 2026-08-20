/*
 * C++ file for the 2D TMz output files: probe CSV and binary snapshots.
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
#include <algorithm>
#include <bit>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    static_assert(std::endian::native == std::endian::little, "snapshot files are written in the host's byte order, little-endian");
    static_assert(sizeof(float) == 4 && sizeof(double) == 8, "snapshot values are 4-byte floats or 8-byte doubles");

    // Helpers private to this file.
    namespace
    {
        // Append the bytes of a value to a buffer.
        template <typename T> void put(std::vector<char> &buffer, const T &value)
        {
            const auto *bytes = reinterpret_cast<const char *>(&value);
            buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
        }

        // Read a value from a buffer at an offset.
        template <typename T> T get(const std::vector<char> &buffer, std::size_t offset)
        {
            T value;
            std::copy_n(buffer.data() + offset, sizeof(T), reinterpret_cast<char *>(&value));
            return value;
        }

        // Widen count values of type T from a buffer into doubles.
        template <typename T> std::vector<double> widen(const std::vector<char> &buffer, std::size_t offset, std::size_t count)
        {
            std::vector<double> values(count);
            for (std::size_t k = 0; k < count; ++k) {
                values[k] = static_cast<double>(get<T>(buffer, offset + k * sizeof(T)));
            }
            return values;
        }

        // Build the error for a snapshot file that cannot be read back.
        std::runtime_error snapshot_error(const std::string &name, std::string_view problem)
        {
            std::ostringstream message;
            message << name;
            message << ": ";
            message << problem;
            return std::runtime_error(message.str());
        }

        // Open a file for writing or throw.
        std::ofstream open_output(const std::filesystem::path &path, std::ios::openmode mode)
        {
            std::ofstream out(path, mode | std::ios::trunc);
            if (!out) {
                std::ostringstream message;
                message << "cannot write ";
                message << path.string();
                throw std::runtime_error(message.str());
            }
            return out;
        }

        // Check that a stream is still good after writing.
        void require_written(const std::ofstream &out, const std::filesystem::path &path)
        {
            if (!out) {
                std::ostringstream message;
                message << "cannot write ";
                message << path.string();
                throw std::runtime_error(message.str());
            }
        }
    }

    // Write the header and Ez row by row, without the padding.
    void write_snapshot(const std::filesystem::path &path, const Fields &fields, std::size_t step, double dt)
    {
        const Grid &grid = fields.grid();
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

    // Read the whole file, check the header against its length and widen Ez.
    SnapshotData read_snapshot(const std::filesystem::path &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::ostringstream message;
            message << "cannot read ";
            message << path.string();
            throw std::runtime_error(message.str());
        }
        const std::vector<char> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string name = path.string();
        if (buffer.size() < SNAPSHOT_HEADER_SIZE || !std::equal(std::begin(SNAPSHOT_MAGIC), std::end(SNAPSHOT_MAGIC), buffer.begin())) {
            throw snapshot_error(name, "not a snapshot file");
        }
        if (get<std::uint32_t>(buffer, 8) != SNAPSHOT_VERSION) {
            throw snapshot_error(name, "unsupported snapshot version");
        }
        const auto size = get<std::uint32_t>(buffer, 12);
        if (size != 4 && size != 8) {
            throw snapshot_error(name, "unsupported value size");
        }
        SnapshotData data{get<std::uint64_t>(buffer, 16),
                          get<std::uint64_t>(buffer, 24),
                          get<std::uint64_t>(buffer, 32),
                          get<double>(buffer, 40),
                          get<double>(buffer, 48),
                          get<double>(buffer, 56),
                          {}};
        const std::size_t count = data.nx * data.ny;
        if (data.ny != 0 && count / data.ny != data.nx) {
            throw snapshot_error(name, "length does not match the header");
        }
        if ((buffer.size() - SNAPSHOT_HEADER_SIZE) / size != count || (buffer.size() - SNAPSHOT_HEADER_SIZE) % size != 0) {
            throw snapshot_error(name, "length does not match the header");
        }
        data.ez = size == 4 ? widen<float>(buffer, SNAPSHOT_HEADER_SIZE, count) : widen<double>(buffer, SNAPSHOT_HEADER_SIZE, count);
        return data;
    }

    // Write the comment, the header and one row per value, with full precision.
    void write_probe_csv(const std::filesystem::path &path, std::string_view comment, std::span<const double> times,
                         std::span<const FloatType> values)
    {
        if (times.size() != values.size()) {
            throw std::invalid_argument("write_probe_csv: times and values differ in length");
        }
        std::ofstream out = open_output(path, std::ios::out);
        out.precision(std::numeric_limits<double>::max_digits10);
        if (!comment.empty()) {
            out << "# ";
            out << comment;
            out << '\n';
        }
        out << "step,time,ez\n";
        for (std::size_t n = 0; n < values.size(); ++n) {
            out << n + 1;
            out << ',';
            out << times[n];
            out << ',';
            out << static_cast<double>(values[n]);
            out << '\n';
        }
        require_written(out, path);
    }
}
