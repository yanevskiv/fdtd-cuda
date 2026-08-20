/*
 * C++ header for the benchmark program.
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

#ifndef FDTD_BENCHMARK_H
#define FDTD_BENCHMARK_H

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
    /** Order in which the samples of the configurations are taken. */
    enum class BenchmarkOrder {
        /** Every sample of a configuration in a row, then the next configuration. */
        Grouped,

        /** One sample of each configuration per round, after every configuration is warmed up. */
        Interleaved,
    };

    /** Settings of one benchmark run, each with a usable default. */
    struct BenchmarkOptions
    {
        /** Cells along x. */
        std::size_t nx = 256;

        /** Cells along y, or zero for nx. */
        std::size_t ny = 0;

        /** Time steps per sample. */
        std::size_t steps = 500;

        /** Samples per configuration. */
        std::size_t repeat = 10;

        /** Seconds of untimed runs before the samples of each configuration, or zero. */
        double warmup = 0.1;

        /** Order of the samples. */
        BenchmarkOrder order = BenchmarkOrder::Grouped;

        /** Steps between the snapshots a sample takes and copies to the host, or zero for none. */
        std::size_t snapshot_every = 0;

        /** Backends to measure in order; none means every compiled backend. */
        std::vector<TMz::BackendType> backends;

        /** Variant names (see TMz::variant_name()) to measure, OpenMP and CUDA ones mixed; none
         * means "default". */
        std::vector<std::string> variants;

        /** Thread counts to measure, read by OpenMP only; none means 1. */
        std::vector<std::size_t> threads;

        /** Outer boundary condition; the PML takes the default settings of Simulation. */
        TMz::Boundary boundary = TMz::Boundary::Pml;

        /** File the CSV is written to, or empty for the output stream. */
        std::filesystem::path output;

        /** Whether to write only the CSV header and stop. */
        bool header_only = false;

        /** Whether to leave the CSV header out. */
        bool no_header = false;

        /** Whether to write only the measured and reported peaks of the machine and stop. */
        bool peaks = false;

        /** Whether to print the usage and stop. */
        bool help = false;

        /** Whether to print the version and stop. */
        bool version = false;
    };

    /** One timed run of the workload. */
    struct BenchmarkSample
    {
        /** Backend the run was on. */
        TMz::BackendType backend;

        /** Canonical variant name in the backend's grammar, "default" for serial. */
        std::string variant;

        /** Threads the run used, 1 for serial and CUDA. */
        std::size_t threads;

        /** Cells along x. */
        std::size_t nx;

        /** Cells along y. */
        std::size_t ny;

        /** Time steps of the run. */
        std::size_t nsteps;

        /** Index of the sample in its configuration, from zero. */
        std::size_t repeat;

        /** Wall time of Simulation::run in seconds. */
        double seconds;

        /** Sum of Ez over the grid after the run, accumulated in double. */
        double checksum;

        /** Modelled bytes moved to and from memory per step (see TMz::StepTraffic). */
        double bytes_per_step;

        /** Modelled floating-point operations per step. */
        double flops_per_step;
    };

    /**
     * Parse the command-line arguments, without the program name.
     *
     * Each option takes its value as the next argument or after '='. "--backend all"
     * adds every compiled backend.
     *
     * @param args Arguments in order.
     * @return The settings.
     * @throws std::invalid_argument On an unknown option, a missing value or a bad value.
     */
    BenchmarkOptions parse_benchmark_options(std::span<const std::string_view> args);

    /**
     * Usage text of the benchmark program.
     *
     * @param program Name to show for the program.
     * @return The text, ending in a newline.
     */
    std::string benchmark_usage(std::string_view program);

    /**
     * Header line of the benchmark CSV.
     *
     * @return The column names, comma-separated, with no newline.
     */
    std::string benchmark_csv_header();

    /**
     * Format one sample as a CSV row.
     *
     * The precision column is that of this build, float or double.
     *
     * @param sample Sample to format.
     * @return The row, with no newline.
     */
    std::string format_benchmark_row(const BenchmarkSample &sample);

    /**
     * Write the peaks of the machine as '# key: value' lines, for the roofline of a report.
     *
     * The peaks are TMz::machine_peaks() of each available backend, over 128 MiB arrays with
     * 10 timed runs each: the OpenMP triad over three such arrays, the CUDA copy over two of
     * twice that. On a machine shared with other processes, every thread can be slower than
     * fewer, which the triad's thread count shows.
     *
     * @param out Stream for the lines.
     * @throws std::runtime_error When a CUDA call fails.
     */
    void write_benchmark_peaks(std::ostream &out);

    /**
     * Measure the workload in every configuration and write a CSV row per sample.
     *
     * The workload is a grid of vacuum at Courant number 0.99 with a Gaussian Ez source
     * 12 steps wide at the centre. A configuration is a backend, a variant and a thread
     * count. A backend runs the variants it reads, or "default" when it reads none; serial reads
     * "default" only. Serial and CUDA run with one thread, OpenMP with each thread count. Each
     * configuration is warmed up with untimed runs, then measured repeat times; every sample
     * builds a new Simulation and times only Simulation::run, which with snapshot_every takes a
     * snapshot every so many steps and hands it to a handler that drops it. Grouped takes the
     * samples of a configuration in a row after its warm-up; interleaved warms every configuration
     * up, then takes one sample of each per round. A summary line per configuration goes to the log
     * once its samples are taken.
     *
     * @param options Settings of the run.
     * @param out Stream for the CSV when no output file is set.
     * @param log Stream for the summary lines.
     * @return The samples in the order written.
     * @throws std::invalid_argument On zero steps or repeats, a backend that is not
     *         compiled in, a variant no backend reads, or settings the simulation rejects.
     * @throws std::runtime_error When the output file cannot be written.
     */
    std::vector<BenchmarkSample> run_benchmark(const BenchmarkOptions &options, std::ostream &out,
                                               std::ostream &log);
}

#endif
