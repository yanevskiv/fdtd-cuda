/*
 * C++ file for the output file tests.
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

#include <catch2/catch_test_macros.hpp>

#include "fdtd/TMz/fdtd_tmz_fields2d.h"
#include "fdtd/TMz/fdtd_tmz_output.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using fdtd::FloatType;
using fdtd::TMz::Fields2D;
using fdtd::TMz::Grid2D;

namespace
{
    // A directory under the system temporary directory, removed on destruction.
    class TempDir
    {
    public:
        // Create a fresh directory named after a test.
        explicit TempDir(const std::string &name)
            : m_path(std::filesystem::temp_directory_path() / ("fdtd_test_" + name))
        {
            std::filesystem::remove_all(m_path);
            std::filesystem::create_directories(m_path);
        }

        // Remove the directory and its files.
        ~TempDir()
        {
            std::error_code ec;
            std::filesystem::remove_all(m_path, ec);
        }

        TempDir(const TempDir &) = delete;
        TempDir &operator=(const TempDir &) = delete;

        // Return the directory.
        const std::filesystem::path &path() const noexcept
        {
            return m_path;
        }

    private:
        std::filesystem::path m_path;
    };

    // Read a whole file as text.
    std::string read_text(const std::filesystem::path &path)
    {
        std::ifstream in(path, std::ios::binary);
        std::ostringstream text;
        text << in.rdbuf();
        return text.str();
    }

    // Write bytes to a file.
    void write_text(const std::filesystem::path &path, const std::string &bytes)
    {
        std::ofstream out(path, std::ios::binary);
        out << bytes;
    }
}

// Verify that a snapshot reads back with its header and Ez, without the row padding.
TEST_CASE("Snapshots round-trip", "[unit][output]")
{
    const TempDir dir("snapshot_round_trip");
    const Grid2D grid(13, 7, FloatType(0.5), FloatType(0.25));
    Fields2D fields(grid);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            fields.ez()[grid.index(i, j)] = FloatType(i) - FloatType(0.5) * FloatType(j);
            fields.hx()[grid.index(i, j)] = FloatType(99);
        }
    }
    const auto path = dir.path() / "snap.bin";
    fdtd::TMz::write_snapshot(path, fields, 42, 1.5e-12);
    CHECK(std::filesystem::file_size(path) ==
          fdtd::TMz::SNAPSHOT_HEADER_SIZE + 13 * 7 * sizeof(FloatType));
    CHECK(read_text(path).substr(0, 8) == "FDTDSNAP");

    const auto data = fdtd::TMz::read_snapshot(path);
    CHECK(data.nx == 13);
    CHECK(data.ny == 7);
    CHECK(data.step == 42);
    CHECK(data.dt == 1.5e-12);
    CHECK(data.dx == 0.5);
    CHECK(data.dy == 0.25);
    REQUIRE(data.ez.size() == 13 * 7);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            CHECK(data.ez[j * 13 + i] == double(fields.ez()[grid.index(i, j)]));
        }
    }
}

// Verify that files that are not whole snapshots are rejected.
TEST_CASE("Snapshot reader rejects bad files", "[unit][output]")
{
    const TempDir dir("snapshot_bad");
    const Grid2D grid(4, 3);
    const auto good = dir.path() / "good.bin";
    fdtd::TMz::write_snapshot(good, Fields2D(grid), 0, 1);
    const std::string bytes = read_text(good);

    const auto bad = dir.path() / "bad.bin";
    write_text(bad, "FDTDSNAX" + bytes.substr(8));
    CHECK_THROWS_AS(fdtd::TMz::read_snapshot(bad), std::runtime_error);
    write_text(bad, bytes.substr(0, bytes.size() - 1));
    CHECK_THROWS_AS(fdtd::TMz::read_snapshot(bad), std::runtime_error);
    write_text(bad, bytes + std::string(sizeof(FloatType), '\0'));
    CHECK_THROWS_AS(fdtd::TMz::read_snapshot(bad), std::runtime_error);
    std::string version = bytes;
    version[8] = 2;
    write_text(bad, version);
    CHECK_THROWS_AS(fdtd::TMz::read_snapshot(bad), std::runtime_error);
    write_text(bad, bytes.substr(0, 20));
    CHECK_THROWS_AS(fdtd::TMz::read_snapshot(bad), std::runtime_error);
    CHECK_THROWS_AS(fdtd::TMz::read_snapshot(dir.path() / "missing.bin"), std::runtime_error);
}

// Verify the probe CSV layout.
TEST_CASE("Probe CSV holds step, time and ez", "[unit][output]")
{
    const TempDir dir("probe_csv");
    const auto path = dir.path() / "probe.csv";
    const std::vector<double> times{0.5, 1.5};
    const std::vector<FloatType> values{FloatType(2), FloatType(-0.25)};
    fdtd::TMz::write_probe_csv(path, "probe 0", times, values);
    CHECK(read_text(path) == "# probe 0\nstep,time,ez\n1,0.5,2\n2,1.5,-0.25\n");

    fdtd::TMz::write_probe_csv(path, "", times, values);
    CHECK(read_text(path).starts_with("step,time,ez\n"));
    CHECK_THROWS_AS(fdtd::TMz::write_probe_csv(path, "", times, {}), std::invalid_argument);
}
