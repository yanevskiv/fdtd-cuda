/*
 * C++ file for the command-line driver tests.
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

#include "fdtd/TMz/fdtd_tmz_output.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/fdtd_driver.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using fdtd::DriverOptions;
using fdtd::parse_driver_options;
using fdtd::TMz::BackendType;
using fdtd::TMz::Boundary;

namespace
{
    // Parse a list of arguments.
    DriverOptions parse(std::vector<std::string_view> args)
    {
        return parse_driver_options(args);
    }
}

// Verify the defaults when no option is given.
TEST_CASE("Driver defaults", "[unit][driver]")
{
    const auto options = parse({});
    CHECK(options.nx == 200);
    CHECK(options.ny == 0);
    CHECK(options.steps == 300);
    CHECK(options.pulses.empty());
    CHECK(options.probes.empty());
    CHECK(options.snapshot_every == 0);
    CHECK(options.boundary == Boundary::Pec);
    const fdtd::TMz::Simulation::Options simDefaults;
    CHECK(options.pml_thickness == simDefaults.pml_thickness);
    CHECK(options.pml_order == simDefaults.pml_order);
    CHECK(options.pml_reflection == simDefaults.pml_reflection);
    CHECK(options.backend == BackendType::Serial);
    CHECK(options.output == "out");
    CHECK_FALSE(options.help);
}

// Verify every option, with the value after '=' or as the next argument.
TEST_CASE("Driver parses every option", "[unit][driver]")
{
    const auto options = parse({"--nx",        "64",
                                "--ny=32",     "--dx",
                                "2e-3",        "--dy=1e-3",
                                "--courant",   "0.5",
                                "--steps=40",  "--pulse",
                                "10,12",       "--pulse=20,22,8",
                                "--probe",     "5,6",
                                "--probe=7,8", "--snapshot-every",
                                "10",          "--boundary=pmc",
                                "--backend",   "serial",
                                "--output",    "dir",
                                "--help",      "-v"});
    CHECK(options.nx == 64);
    CHECK(options.ny == 32);
    CHECK(options.dx == 2e-3);
    CHECK(options.dy == 1e-3);
    CHECK(options.courant == 0.5);
    CHECK(options.steps == 40);
    REQUIRE(options.pulses.size() == 2);
    CHECK(options.pulses[0].i == 10);
    CHECK(options.pulses[0].j == 12);
    CHECK(options.pulses[0].width == 30);
    CHECK(options.pulses[1].width == 8);
    REQUIRE(options.probes.size() == 2);
    CHECK(options.probes[1].i == 7);
    CHECK(options.probes[1].j == 8);
    CHECK(options.snapshot_every == 10);
    CHECK(options.boundary == Boundary::Pmc);
    CHECK(options.output == "dir");
    CHECK(options.help);
    CHECK(options.version);
}

// Verify the PML boundary and its settings.
TEST_CASE("Driver parses the PML options", "[unit][driver]")
{
    const auto options = parse(
        {"--boundary", "pml", "--pml-thickness", "8", "--pml-order=2", "--pml-reflection", "1e-4"});
    CHECK(options.boundary == Boundary::Pml);
    CHECK(options.pml_thickness == 8);
    CHECK(options.pml_order == 2);
    CHECK(options.pml_reflection == 1e-4);
    CHECK(parse({"--pml-thickness=0"}).pml_thickness == 0);
}

// Verify that bad arguments throw.
TEST_CASE("Driver rejects bad arguments", "[unit][driver]")
{
    CHECK_THROWS_AS(parse({"--nx"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--nx", "ten"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--nx=-3"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--dx", "0"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--courant=nan"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--pulse", "1"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--pulse", "1,2,3,4"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--probe", "1,2,3"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--boundary", "abc"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--boundary", "PML"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--pml-thickness", "-1"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--pml-order", "0"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--pml-reflection=nan"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--backend", "gpu"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--output="}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--frobnicate", "1"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"stray"}), std::invalid_argument);
}

// Verify that a small run writes its files and that the last snapshot agrees with the probe.
TEST_CASE("Driver smoke run", "[unit][driver]")
{
    const auto dir = std::filesystem::temp_directory_path() / "fdtd_test_driver_smoke";
    std::filesystem::remove_all(dir);
    auto options = parse({"--nx", "32", "--ny", "24", "--steps", "25", "--probe", "20,12",
                          "--snapshot-every", "10", "--boundary", "pmc"});
    options.output = dir;
    std::ostringstream log;
    fdtd::run_driver(options, log);

    for (const char *name : {"snapshot_000000.bin", "snapshot_000010.bin", "snapshot_000020.bin",
                             "snapshot_000025.bin", "probe_0.csv"}) {
        CAPTURE(name);
        CHECK(std::filesystem::exists(dir / name));
    }
    const auto last = fdtd::TMz::read_snapshot(dir / "snapshot_000025.bin");
    CHECK(last.nx == 32);
    CHECK(last.ny == 24);
    CHECK(last.step == 25);

    std::ifstream csv(dir / "probe_0.csv");
    std::string line;
    std::string previous;
    while (std::getline(csv, line)) {
        previous = line;
    }
    const std::string value = previous.substr(previous.rfind(',') + 1);
    CHECK(std::stod(value) == last.ez[12 * 32 + 20]);
    CHECK(std::stod(value) != 0);
    CHECK(log.str().find("probe_0.csv") != std::string::npos);
    std::filesystem::remove_all(dir);
}

// Verify that a PML run writes its files, keeps the wall at zero and logs the boundary.
TEST_CASE("Driver smoke run with the PML", "[unit][driver]")
{
    const auto dir = std::filesystem::temp_directory_path() / "fdtd_test_driver_pml";
    std::filesystem::remove_all(dir);
    auto options = parse({"--nx", "48", "--ny", "40", "--steps", "60", "--probe", "30,20",
                          "--snapshot-every", "30", "--boundary", "pml", "--pml-thickness", "8"});
    options.output = dir;
    std::ostringstream log;
    fdtd::run_driver(options, log);

    CHECK(std::filesystem::exists(dir / "probe_0.csv"));
    const auto last = fdtd::TMz::read_snapshot(dir / "snapshot_000060.bin");
    REQUIRE(last.nx == 48);
    REQUIRE(last.ny == 40);
    bool edgeZero = true;
    for (std::size_t i = 0; i < last.nx; ++i) {
        edgeZero = edgeZero && last.ez[i] == 0 && last.ez[(last.ny - 1) * last.nx + i] == 0;
    }
    for (std::size_t j = 0; j < last.ny; ++j) {
        edgeZero = edgeZero && last.ez[j * last.nx] == 0 && last.ez[j * last.nx + last.nx - 1] == 0;
    }
    CHECK(edgeZero);
    CHECK(last.ez[20 * 48 + 24] != 0);
    CHECK(log.str().find("boundary pml (8 cells)") != std::string::npos);
    std::filesystem::remove_all(dir);
}

// Verify that the run rejects PML settings the simulation refuses.
TEST_CASE("Driver rejects bad PML settings", "[unit][driver]")
{
    const auto dir = std::filesystem::temp_directory_path() / "fdtd_test_driver_pml_bad";
    std::ostringstream log;
    auto reflection = parse({"--nx", "32", "--steps", "1", "--boundary", "pml", "--pml-thickness",
                             "4", "--pml-reflection", "2"});
    reflection.output = dir;
    CHECK_THROWS_AS(fdtd::run_driver(reflection, log), std::invalid_argument);
    auto inside = parse({"--nx", "32", "--steps", "1", "--boundary", "pml", "--pml-thickness", "4",
                         "--pulse", "2,16"});
    inside.output = dir;
    CHECK_THROWS_AS(fdtd::run_driver(inside, log), std::invalid_argument);
    auto thick =
        parse({"--nx", "32", "--steps", "1", "--boundary", "pml", "--pml-thickness", "16"});
    thick.output = dir;
    CHECK_THROWS_AS(fdtd::run_driver(thick, log), std::invalid_argument);
    std::filesystem::remove_all(dir);
}
