/*
 * C++ header for the 2D TMz output files: probe CSV and binary snapshots.
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

#ifndef FDTD_TMZ_OUTPUT_H
#define FDTD_TMZ_OUTPUT_H

#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** First eight bytes of a snapshot file. */
        inline constexpr char SNAPSHOT_MAGIC[8] = {'F', 'D', 'T', 'D', 'S', 'N', 'A', 'P'};

        /** Version of the snapshot layout written by write_snapshot(). */
        inline constexpr std::uint32_t SNAPSHOT_VERSION = 1;

        /** Size of the snapshot header in bytes; Ez follows it. */
        inline constexpr std::size_t SNAPSHOT_HEADER_SIZE = 64;

        /** Contents of a snapshot file, with Ez widened to double. */
        struct SnapshotData
        {
            /** Cells along x. */
            std::size_t nx;

            /** Cells along y. */
            std::size_t ny;

            /** Steps run when the snapshot was taken. */
            std::size_t step;

            /** Time step in seconds. */
            double dt;

            /** Cell size along x in meters. */
            double dx;

            /** Cell size along y in meters. */
            double dy;

            /** Ez, nx * ny values, row by row with x fastest. */
            std::vector<double> ez;
        };

        /**
         * Write Ez to a binary snapshot file.
         *
         * The file is little-endian with no padding: the magic "FDTDSNAP", then uint32 version and uint32 sizeof(FloatType), uint64 nx, ny
         * and step, float64 dt, dx and dy, 64 bytes in all, then the nx * ny Ez values as FloatType, row by row with x fastest. Ez is at
         * step * dt.
         *
         * @param path File to create or replace.
         * @param fields Fields to write Ez from.
         * @param step Steps run when the fields were taken.
         * @param dt Time step in seconds.
         * @throws std::runtime_error When the file cannot be written.
         */
        void write_snapshot(const std::filesystem::path &path, const Fields &fields, std::size_t step, double dt);

        /**
         * Read a snapshot file written with float or double values.
         *
         * @param path File to read.
         * @return The contents.
         * @throws std::runtime_error When the file cannot be read, has another magic or version, a value size other than 4 or 8, or a
         *         length that does not match its header.
         */
        SnapshotData read_snapshot(const std::filesystem::path &path);

        /**
         * Write a probe's record as CSV with the columns step, time and ez.
         *
         * Row n holds step n + 1, the time of the value and the value. Lines that start with '#' before the header describe the probe.
         *
         * @param path File to create or replace.
         * @param comment Text of the leading comment line, or empty for none.
         * @param times Time of each value in seconds.
         * @param values Recorded values, as many as times.
         * @throws std::invalid_argument When times and values differ in length.
         * @throws std::runtime_error When the file cannot be written.
         */
        void write_probe_csv(const std::filesystem::path &path, std::string_view comment, std::span<const double> times,
                             std::span<const FloatType> values);
    }
}

#endif
