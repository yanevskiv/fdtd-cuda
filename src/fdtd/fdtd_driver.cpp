/*
 * C++ file for the fdtd command-line driver.
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
#include "fdtd/fdtd_driver.h"

// The simulation, its output files and the source waveform.
#include "fdtd/TMz/fdtd_tmz_output.h"
#include "fdtd/TMz/fdtd_tmz_sim2d.h"
#include "fdtd/fdtd_waveform.h"

// Standard library for parsing, formatting, math and errors.
#include <charconv>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <system_error>

// Namespace for the command-line driver.
namespace fdtd::driver
{
    // Helpers private to this file.
    namespace
    {
        // Parse a whole non-negative integer.
        std::size_t parse_size(std::string_view name, std::string_view text)
        {
            std::size_t value = 0;
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc() || end != text.data() + text.size()) {
                throw std::invalid_argument(std::string(name) +
                                            ": not a whole number: " + std::string(text));
            }
            return value;
        }

        // Parse a whole positive finite number.
        double parse_positive(std::string_view name, std::string_view text)
        {
            double value = 0;
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc() || end != text.data() + text.size() || !(value > 0) ||
                !std::isfinite(value)) {
                throw std::invalid_argument(std::string(name) +
                                            ": not a positive number: " + std::string(text));
            }
            return value;
        }

        // Split comma-separated fields.
        std::vector<std::string_view> split(std::string_view text)
        {
            std::vector<std::string_view> fields;
            std::size_t start = 0;
            while (true) {
                const std::size_t comma = text.find(',', start);
                fields.push_back(text.substr(start, comma - start));
                if (comma == std::string_view::npos) {
                    return fields;
                }
                start = comma + 1;
            }
        }

        // Parse I,J[,WIDTH] for a pulse.
        PulseSpec parse_pulse(std::string_view text)
        {
            const auto fields = split(text);
            if (fields.size() < 2 || fields.size() > 3) {
                throw std::invalid_argument("--pulse: expected I,J[,WIDTH]: " + std::string(text));
            }
            return {parse_size("--pulse", fields[0]), parse_size("--pulse", fields[1]),
                    fields.size() == 3 ? parse_positive("--pulse", fields[2]) : 30.0};
        }

        // Parse I,J for a probe.
        ProbeSpec parse_probe(std::string_view text)
        {
            const auto fields = split(text);
            if (fields.size() != 2) {
                throw std::invalid_argument("--probe: expected I,J: " + std::string(text));
            }
            return {parse_size("--probe", fields[0]), parse_size("--probe", fields[1])};
        }

        // Parse a boundary name.
        TMz::Boundary parse_boundary(std::string_view text)
        {
            if (text == "pec") {
                return TMz::Boundary::Pec;
            }
            if (text == "pmc") {
                return TMz::Boundary::Pmc;
            }
            throw std::invalid_argument("--boundary: expected pec or pmc: " + std::string(text));
        }

        // Parse a backend name against every backend type.
        TMz::BackendType parse_backend(std::string_view text)
        {
            for (const auto type :
                 {TMz::BackendType::Serial, TMz::BackendType::OpenMP, TMz::BackendType::Cuda}) {
                if (text == TMz::backend_name(type)) {
                    return type;
                }
            }
            throw std::invalid_argument("--backend: expected serial, openmp or cuda: " +
                                        std::string(text));
        }

        // Apply one option and its value to the settings.
        void apply(DriverOptions &options, std::string_view name, std::string_view value)
        {
            if (name == "--nx") {
                options.nx = parse_size(name, value);
            } else if (name == "--ny") {
                options.ny = parse_size(name, value);
            } else if (name == "--dx") {
                options.dx = parse_positive(name, value);
            } else if (name == "--dy") {
                options.dy = parse_positive(name, value);
            } else if (name == "--courant") {
                options.courant = parse_positive(name, value);
            } else if (name == "--steps") {
                options.steps = parse_size(name, value);
            } else if (name == "--pulse") {
                options.pulses.push_back(parse_pulse(value));
            } else if (name == "--probe") {
                options.probes.push_back(parse_probe(value));
            } else if (name == "--snapshot-every") {
                options.snapshot_every = parse_size(name, value);
            } else if (name == "--boundary") {
                options.boundary = parse_boundary(value);
            } else if (name == "--backend") {
                options.backend = parse_backend(value);
            } else if (name == "--output") {
                if (value.empty()) {
                    throw std::invalid_argument("--output: empty directory");
                }
                options.output = std::string(value);
            } else {
                throw std::invalid_argument("unknown option: " + std::string(name));
            }
        }

        // Format a step as a fixed-width file name part.
        std::string step_name(std::size_t step)
        {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%06zu", step);
            return buffer;
        }

        // Write a snapshot of the current fields and log it.
        void write_current(const TMz::Sim2D &sim, const std::filesystem::path &dir,
                           std::ostream &log)
        {
            const auto snap = sim.snapshot();
            const auto path = dir / ("snapshot_" + step_name(snap.step) + ".bin");
            TMz::write_snapshot(path, snap.fields, snap.step, sim.dt());
            log << "wrote " << path.string() << '\n';
        }
    }

    // Walk the arguments, splitting "--name=value" or taking the next argument as the value.
    DriverOptions parse_options(std::span<const std::string_view> args)
    {
        DriverOptions options;
        for (std::size_t k = 0; k < args.size(); ++k) {
            const std::string_view arg = args[k];
            if (arg == "-h" || arg == "--help") {
                options.help = true;
                continue;
            }
            if (arg == "-v" || arg == "--version") {
                options.version = true;
                continue;
            }
            if (!arg.starts_with("--")) {
                throw std::invalid_argument("unexpected argument: " + std::string(arg));
            }
            const std::size_t equals = arg.find('=');
            if (equals != std::string_view::npos) {
                apply(options, arg.substr(0, equals), arg.substr(equals + 1));
                continue;
            }
            if (k + 1 == args.size()) {
                throw std::invalid_argument(std::string(arg) + ": missing value");
            }
            apply(options, arg, args[++k]);
        }
        return options;
    }

    // Build the usage text.
    std::string usage(std::string_view program)
    {
        return "Usage: " + std::string(program) +
               " [options]\n"
               "\n"
               "2D TMz FDTD simulation. Runs soft Gaussian Ez sources on a grid and writes\n"
               "probe traces as CSV and Ez snapshots as binary files.\n"
               "\n"
               "Options:\n"
               "  --nx N                Cells along x (default: 200)\n"
               "  --ny N                Cells along y (default: nx)\n"
               "  --dx M                Cell size along x in meters (default: 1e-3)\n"
               "  --dy M                Cell size along y in meters (default: dx)\n"
               "  --courant S           Courant number in (0, 1] (default: 0.99)\n"
               "  --steps N             Time steps to run (default: 300)\n"
               "  --pulse I,J[,W]       Gaussian Ez source W steps wide (default W: 30);\n"
               "                        repeatable; none means one at the centre\n"
               "  --probe I,J           Record Ez at a cell into probe_<k>.csv; repeatable\n"
               "  --snapshot-every N    Write snapshot_<step>.bin every N steps (default: 0,\n"
               "                        none)\n"
               "  --boundary B          Outer boundary, pec or pmc (default: pec)\n"
               "  --backend B           Backend, serial, openmp or cuda (default: serial)\n"
               "  --output DIR          Directory for the files (default: out)\n"
               "  -h, --help            Show this message and exit\n"
               "  -v, --version         Show the version and exit\n"
               "\n"
               "Example:\n"
               "  " +
               std::string(program) +
               " --nx 256 --steps 400 --probe 160,128 --snapshot-every 50 --output out\n";
    }

    // Build the simulation, run it in pieces between snapshots, then write the probes.
    void run(const DriverOptions &options, std::ostream &log)
    {
        TMz::Sim2D::Options simOptions;
        simOptions.dx = FloatType(options.dx);
        simOptions.dy = FloatType(options.dy > 0 ? options.dy : options.dx);
        simOptions.courant = options.courant;
        simOptions.boundary = options.boundary;
        simOptions.backend = options.backend;
        const std::size_t ny = options.ny > 0 ? options.ny : options.nx;
        TMz::Sim2D sim(options.nx, ny, simOptions);

        if (options.pulses.empty()) {
            sim.add_source(options.nx / 2, ny / 2, TMz::Component::Ez,
                           Gaussian::from_steps(sim.dt(), 30));
        }
        for (const PulseSpec &pulse : options.pulses) {
            sim.add_source(pulse.i, pulse.j, TMz::Component::Ez,
                           Gaussian::from_steps(sim.dt(), pulse.width));
        }
        for (const ProbeSpec &probe : options.probes) {
            sim.add_probe(probe.i, probe.j, TMz::Component::Ez);
        }

        std::filesystem::create_directories(options.output);
        log << "grid " << options.nx << " x " << ny << ", dt " << sim.dt() << " s, "
            << options.steps << " steps on " << TMz::backend_name(options.backend) << '\n';

        const std::size_t every = options.snapshot_every;
        if (every > 0) {
            write_current(sim, options.output, log);
        }
        while (sim.step_count() < options.steps) {
            const std::size_t left = options.steps - sim.step_count();
            sim.run(every > 0 && every < left ? every : left);
            if (every > 0) {
                write_current(sim, options.output, log);
            }
        }

        for (std::size_t k = 0; k < options.probes.size(); ++k) {
            const ProbeSpec &probe = options.probes[k];
            const auto path = options.output / ("probe_" + std::to_string(k) + ".csv");
            const std::string comment = "probe " + std::to_string(k) +
                                        " at i=" + std::to_string(probe.i) +
                                        ", j=" + std::to_string(probe.j) + ", component Ez";
            TMz::write_probe_csv(path, comment, sim.probe_times(k), sim.probe_data(k));
            log << "wrote " << path.string() << '\n';
        }
    }
}
