/*
 * C++ header for the fdtd command-line driver.
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

#ifndef FDTD_DRIVER_H
#define FDTD_DRIVER_H

#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"

#include <cstddef>
#include <filesystem>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** A soft Gaussian Ez source given on the command line. */
    struct DriverPulseSpec
    {
        /** Column of the cell. */
        std::size_t i;

        /** Row of the cell. */
        std::size_t j;

        /** Width of the Gaussian in time steps. */
        double width;
    };

    /** A probe cell given on the command line. */
    struct DriverProbeSpec
    {
        /** Column of the cell. */
        std::size_t i;

        /** Row of the cell. */
        std::size_t j;
    };

    /** Settings of one driver run, each with a usable default. */
    struct DriverOptions
    {
        /** Cells along x. */
        std::size_t nx = 200;

        /** Cells along y, or zero for nx. */
        std::size_t ny = 0;

        /** Cell size along x in meters. */
        double dx = 1e-3;

        /** Cell size along y in meters, or zero for dx. */
        double dy = 0;

        /** Courant number. */
        double courant = 0.99;

        /** Time steps to run. */
        std::size_t steps = 300;

        /** Gaussian sources; none means one of width 30 at the centre. */
        std::vector<DriverPulseSpec> pulses;

        /** Probe cells. */
        std::vector<DriverProbeSpec> probes;

        /** Steps between snapshots, or zero for none. */
        std::size_t snapshot_every = 0;

        /** Outer boundary condition. */
        TMz::Boundary boundary = TMz::Boundary::Pec;

        /** PML thickness in cells, used with the PML boundary; zero turns the PML off. */
        std::size_t pml_thickness = 20;

        /** Grading exponent of the PML conductivity. */
        double pml_order = 3;

        /** Target reflection of the PML at normal incidence, in (0, 1). */
        double pml_reflection = 1e-6;

        /** Backend to run on. */
        TMz::BackendType backend = TMz::BackendType::Serial;

        /** Thread count of the OpenMP backend, 0 for the OpenMP default. */
        std::size_t threads = 0;

        /** Variant name of the backend, as TMz::variant_name() reads it. */
        std::string variant = "default";

        /** Directory the files are written to. */
        std::filesystem::path output = "out";

        /** Whether to print the usage and stop. */
        bool help = false;

        /** Whether to print the version and stop. */
        bool version = false;
    };

    /**
     * Parse the command-line arguments, without the program name.
     *
     * Each option takes its value as the next argument or after '='. The variant is checked
     * against the backend once every option is read, so the two may come in either order.
     *
     * @param args Arguments in order.
     * @return The settings.
     * @throws std::invalid_argument On an unknown option, a missing value, a bad value, or a
     *         variant the backend does not read.
     */
    DriverOptions parse_driver_options(std::span<const std::string_view> args);

    /**
     * Usage text of the driver.
     *
     * @param program Name to show for the program.
     * @return The text, ending in a newline.
     */
    std::string driver_usage(std::string_view program);

    /**
     * Run a simulation and write its probe CSV files and snapshots.
     *
     * Writes probe_<k>.csv per probe and, with snapshot_every, snapshot_<step>.bin at
     * step zero, every snapshot_every steps and the last step.
     *
     * @param options Settings of the run.
     * @param log Stream for progress lines.
     * @throws std::invalid_argument On settings the simulation rejects.
     * @throws std::runtime_error When a file cannot be written.
     */
    void run_driver(const DriverOptions &options, std::ostream &log);
}

#endif
