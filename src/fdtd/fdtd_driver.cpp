/*
 * C++ file for the fdtd command-line driver.
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

// Shared argument parsing.
#include "fdtd/fdtd_args.h"

// The simulation, its output files, the backend variants and the source waveform.
#include "fdtd/TMz/fdtd_tmz_output.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"

// Standard library for string building and errors.
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

// Namespace for the fdtd library.
namespace fdtd
{
    // Helpers private to this file.
    namespace
    {
        // Parse I,J[,WIDTH] for a pulse.
        DriverPulseSpec parse_pulse(std::string_view text)
        {
            const auto fields = split_list_arg(text);
            if (fields.size() < 2 || fields.size() > 3) {
                std::ostringstream message;
                message << "--pulse: expected I,J[,WIDTH]: ";
                message << text;
                throw std::invalid_argument(message.str());
            }
            return {
                .i = parse_size_arg("--pulse", fields[0]),
                .j = parse_size_arg("--pulse", fields[1]),
                .width = fields.size() == 3 ? parse_positive_arg("--pulse", fields[2]) : 30.0,
            };
        }

        // Parse I,J for a probe.
        DriverProbeSpec parse_probe(std::string_view text)
        {
            const auto fields = split_list_arg(text);
            if (fields.size() != 2) {
                std::ostringstream message;
                message << "--probe: expected I,J: ";
                message << text;
                throw std::invalid_argument(message.str());
            }
            return {
                .i = parse_size_arg("--probe", fields[0]),
                .j = parse_size_arg("--probe", fields[1]),
            };
        }

        // Describe the boundary for the log, with the PML thickness when it has one.
        std::string describe_boundary(const DriverOptions &options)
        {
            switch (options.boundary) {
                case TMz::Boundary::Pec:
                    return "pec";
                case TMz::Boundary::Pmc:
                    return "pmc";
                case TMz::Boundary::Pml: {
                    std::ostringstream text;
                    text << "pml (";
                    text << options.pml_thickness;
                    text << " cells)";
                    return text.str();
                }
            }
            return "unknown";
        }

        // Apply one option and its value to the settings.
        void apply(DriverOptions &options, std::string_view name, std::string_view value)
        {
            if (name == "--nx") {
                options.nx = parse_size_arg(name, value);
            } else if (name == "--ny") {
                options.ny = parse_size_arg(name, value);
            } else if (name == "--dx") {
                options.dx = parse_positive_arg(name, value);
            } else if (name == "--dy") {
                options.dy = parse_positive_arg(name, value);
            } else if (name == "--courant") {
                options.courant = parse_positive_arg(name, value);
            } else if (name == "--steps") {
                options.steps = parse_size_arg(name, value);
            } else if (name == "--pulse") {
                options.pulses.push_back(parse_pulse(value));
            } else if (name == "--probe") {
                options.probes.push_back(parse_probe(value));
            } else if (name == "--snapshot-every") {
                options.snapshot_every = parse_size_arg(name, value);
            } else if (name == "--boundary") {
                options.boundary = parse_boundary_arg(name, value);
            } else if (name == "--pml-thickness") {
                options.pml_thickness = parse_size_arg(name, value);
            } else if (name == "--pml-order") {
                options.pml_order = parse_positive_arg(name, value);
            } else if (name == "--pml-reflection") {
                options.pml_reflection = parse_positive_arg(name, value);
            } else if (name == "--backend") {
                options.backend = parse_backend_arg(name, value);
            } else if (name == "--threads") {
                options.threads = parse_size_arg(name, value);
            } else if (name == "--variant") {
                options.variant = std::string(value);
            } else if (name == "--output") {
                if (value.empty()) {
                    throw std::invalid_argument("--output: empty directory");
                }
                options.output = std::string(value);
            } else {
                std::ostringstream message;
                message << "unknown option: ";
                message << name;
                throw std::invalid_argument(message.str());
            }
        }

        // Name the snapshot file of a step, the step zero-padded to six digits.
        std::string snapshot_name(std::size_t step)
        {
            std::ostringstream name;
            name << "snapshot_";
            name << std::setw(6);
            name << std::setfill('0');
            name << step;
            name << ".bin";
            return name.str();
        }

        // Write the snapshot of a step and log it.
        void write_fields(const std::filesystem::path &dir, std::size_t step,
                          const TMz::Fields &fields, double dt, std::ostream &log)
        {
            const auto path = dir / snapshot_name(step);
            TMz::write_snapshot(path, fields, step, dt);
            log << "wrote ";
            log << path.string();
            log << '\n';
        }
    }

    // Split the arguments, then set the flags and apply each option with its value.
    DriverOptions parse_driver_options(std::span<const std::string_view> args)
    {
        static constexpr std::string_view FLAGS[] = {"-h", "--help", "-v", "--version"};
        DriverOptions options;
        for (const Argument &arg : split_arguments(args, FLAGS)) {
            if (arg.name == "-h" || arg.name == "--help") {
                options.help = true;
            } else if (arg.name == "-v" || arg.name == "--version") {
                options.version = true;
            } else {
                apply(options, arg.name, arg.value);
            }
        }
        TMz::Simulation::Options check;
        parse_variant_arg("--variant", options.backend, options.variant, check);
        return options;
    }

    // Build the usage text.
    std::string driver_usage(std::string_view program)
    {
        std::ostringstream usage;
        usage << "Usage: ";
        usage << program;
        usage << " [options]\n";
        usage << "\n";
        usage << "2D TMz FDTD simulation. Runs soft Gaussian Ez sources on a grid and writes\n";
        usage << "probe traces as CSV and Ez snapshots as binary files.\n";
        usage << "\n";
        usage << "Options:\n";
        usage << "  --nx N                Cells along x (default: 200)\n";
        usage << "  --ny N                Cells along y (default: nx)\n";
        usage << "  --dx M                Cell size along x in meters (default: 1e-3)\n";
        usage << "  --dy M                Cell size along y in meters (default: dx)\n";
        usage << "  --courant S           Courant number in (0, 1] (default: 0.99)\n";
        usage << "  --steps N             Time steps to run (default: 300)\n";
        usage << "  --pulse I,J[,W]       Gaussian Ez source W steps wide (default W: 30);\n";
        usage << "                        repeatable; none means one at the centre\n";
        usage << "  --probe I,J           Record Ez at a cell into probe_<k>.csv; repeatable\n";
        usage << "  --snapshot-every N    Write snapshot_<step>.bin every N steps (default: 0,\n";
        usage << "                        none)\n";
        usage << "  --boundary B          Outer boundary, pec, pmc or pml (default: pec)\n";
        usage << "  --pml-thickness N     PML thickness in cells, 0 for none (default: 20)\n";
        usage << "  --pml-order M         Grading exponent of the PML (default: 3)\n";
        usage << "  --pml-reflection R    Target PML reflection at normal incidence, in\n";
        usage << "                        (0, 1) (default: 1e-6)\n";
        usage << "  --backend B           Backend, serial, openmp or cuda (default: serial)\n";
        usage << "  --threads N           OpenMP threads, 0 for the OpenMP default (default: 0)\n";
        usage << "  --variant V           Variant of the backend: for OpenMP\n";
        usage << "                        SCHEDULE[:WxH][+simd][+phase], SCHEDULE rows,\n";
        usage << "                        collapse2, tiles or tasks; for CUDA\n";
        usage << "                        KERNEL[:BXxBY][+table][+graph][+async], KERNEL\n";
        usage << "                        fused, split or tiled (default: default)\n";
        usage << "  --output DIR          Directory for the files (default: out)\n";
        usage << "  -h, --help            Show this message and exit\n";
        usage << "  -v, --version         Show the version and exit\n";
        usage << "\n";
        usage << "Example:\n";
        usage << "  ";
        usage << program;
        usage << " --nx 256 --steps 400 --probe 160,128 --snapshot-every 50 --output out\n";
        return usage.str();
    }

    // Build the simulation, run it, writing the snapshots as they come, then write the probes.
    void run_driver(const DriverOptions &options, std::ostream &log)
    {
        TMz::Simulation::Options simOptions;
        simOptions.dx = FloatType(options.dx);
        simOptions.dy = FloatType(options.dy > 0 ? options.dy : options.dx);
        simOptions.courant = options.courant;
        simOptions.boundary = options.boundary;
        simOptions.pml_thickness = options.pml_thickness;
        simOptions.pml_order = options.pml_order;
        simOptions.pml_reflection = options.pml_reflection;
        simOptions.backend = options.backend;
        parse_variant_arg("--variant", options.backend, options.variant, simOptions);
        simOptions.omp_threads = options.threads;
        const std::size_t ny = options.ny > 0 ? options.ny : options.nx;
        TMz::Simulation sim(options.nx, ny, simOptions);

        if (options.pulses.empty()) {
            sim.add_source(options.nx / 2, ny / 2, TMz::Component::Ez,
                           TMz::Gaussian::from_steps(sim.dt(), 30));
        }
        for (const DriverPulseSpec &pulse : options.pulses) {
            sim.add_source(pulse.i, pulse.j, TMz::Component::Ez,
                           TMz::Gaussian::from_steps(sim.dt(), pulse.width));
        }
        for (const DriverProbeSpec &probe : options.probes) {
            sim.add_probe(probe.i, probe.j, TMz::Component::Ez);
        }

        std::filesystem::create_directories(options.output);
        log << "grid ";
        log << options.nx;
        log << " x ";
        log << ny;
        log << ", dt ";
        log << sim.dt();
        log << " s, ";
        log << options.steps;
        log << " steps on ";
        log << TMz::backend_name(options.backend);
        if (options.backend != TMz::BackendType::Serial) {
            log << " (";
            log << variant_name(options.backend, simOptions);
            if (options.backend == TMz::BackendType::OpenMP && options.threads > 0) {
                log << ", ";
                log << options.threads;
                log << " threads";
            }
            log << ")";
        }
        log << ", boundary ";
        log << describe_boundary(options);
        log << '\n';

        const std::size_t every = options.snapshot_every;
        if (every > 0) {
            const auto first = sim.snapshot();
            write_fields(options.output, first.step, first.fields, sim.dt(), log);
            for (std::size_t step = every; step < options.steps; step += every) {
                sim.add_snapshot(step);
            }
            sim.add_snapshot(options.steps);
        }
        sim.run(options.steps, [&](std::size_t step, const TMz::Fields &fields) {
            write_fields(options.output, step, fields, sim.dt(), log);
        });

        for (std::size_t k = 0; k < options.probes.size(); ++k) {
            const DriverProbeSpec &probe = options.probes[k];
            std::ostringstream name;
            name << "probe_";
            name << k;
            name << ".csv";
            const auto path = options.output / name.str();
            std::ostringstream comment;
            comment << "probe ";
            comment << k;
            comment << " at i=";
            comment << probe.i;
            comment << ", j=";
            comment << probe.j;
            comment << ", component Ez";
            TMz::write_probe_csv(path, comment.str(), sim.probe_times(k), sim.probe_data(k));
            log << "wrote ";
            log << path.string();
            log << '\n';
        }
    }
}
