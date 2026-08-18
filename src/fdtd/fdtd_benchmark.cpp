/*
 * C++ file for the benchmark program.
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
#include "fdtd/fdtd_benchmark.h"

// Shared argument parsing.
#include "fdtd/fdtd_args.h"

// The simulation, the backend variants and the source waveform.
#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"
#include "fdtd/fdtd_float.h"

// The machine's peaks: the CPU triad and the GPU's queries and copy.
#if FDTD_WITH_OPENMP
#include "fdtd/TMz/backends/fdtd_tmz_backend_omp.h"
#endif
#if FDTD_WITH_CUDA
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_device.h"
#endif

// Standard library for timing, sorting, string building, files and errors.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

// Namespace for the fdtd library.
namespace fdtd
{
    // Helpers private to this file.
    namespace
    {
        // Width of the Gaussian source in time steps, as fdtd-openmp's benchmark uses.
        constexpr double PULSE_WIDTH_STEPS = 12;

        // Precision of this build as the CSV names it.
        constexpr std::string_view PRECISION = FDTD_DOUBLE ? "double" : "float";

        // Bytes of each array the peak measurements move.
        constexpr std::size_t PEAK_BYTES = std::size_t(1) << 27;

        // Timed runs of each peak measurement.
        constexpr std::size_t PEAK_REPEATS = 10;

        // Write one '# key: value unit' line, the value with the given decimals.
        void peak_line(std::ostream &out, std::string_view key, double value, int decimals,
                       std::string_view unit)
        {
            std::ostringstream line;
            line << "# ";
            line << key;
            line << ": ";
            line << std::fixed;
            line << std::setprecision(decimals);
            line << value;
            if (!unit.empty()) {
                line << ' ';
                line << unit;
            }
            line << '\n';
            out << line.str();
        }

#if FDTD_WITH_CUDA

        // FP32 and FP64 lanes per multiprocessor of a compute capability, zero when not known.
        std::pair<int, int> lanes_per_sm(int major, int minor)
        {
            if (major == 6) {
                return minor == 0 ? std::pair{64, 32} : std::pair{128, 4};
            }
            if (major == 7) {
                return minor == 5 ? std::pair{64, 2} : std::pair{64, 32};
            }
            if (major == 8) {
                return minor == 0 ? std::pair{64, 32} : std::pair{128, 2};
            }
            if (major == 9) {
                return {128, 64};
            }
            return {0, 0};
        }
#endif

        // Add a backend to the list unless it is there already.
        void add_backend(std::vector<TMz::BackendType> &backends, TMz::BackendType type)
        {
            if (std::find(backends.begin(), backends.end(), type) == backends.end()) {
                backends.push_back(type);
            }
        }

        // Read the sample order by name.
        BenchmarkOrder parse_order(std::string_view name, std::string_view value)
        {
            if (value == "grouped") {
                return BenchmarkOrder::Grouped;
            }
            if (value == "interleaved") {
                return BenchmarkOrder::Interleaved;
            }
            std::ostringstream message;
            message << name;
            message << ": expected grouped or interleaved, got '";
            message << value;
            message << "'";
            throw std::invalid_argument(message.str());
        }

        // A backend, a variant and a thread count, measured together.
        struct Configuration
        {
            // Backend to run on.
            TMz::BackendType backend;

            // Variant name as given.
            std::string variant;

            // Thread count, 1 for serial and CUDA.
            std::size_t threads;
        };

        // Apply one option and its value to the settings.
        void apply(BenchmarkOptions &options, std::string_view name, std::string_view value)
        {
            if (name == "--nx") {
                options.nx = parse_size_arg(name, value);
            } else if (name == "--ny") {
                options.ny = parse_size_arg(name, value);
            } else if (name == "--steps") {
                options.steps = parse_size_arg(name, value);
            } else if (name == "--repeat") {
                options.repeat = parse_size_arg(name, value);
            } else if (name == "--warmup") {
                options.warmup = parse_nonnegative_arg(name, value);
            } else if (name == "--order") {
                options.order = parse_order(name, value);
            } else if (name == "--backend") {
                if (value == "all") {
                    for (const auto type : TMz::available_backends()) {
                        add_backend(options.backends, type);
                    }
                } else {
                    add_backend(options.backends, parse_backend_arg(name, value));
                }
            } else if (name == "--variant") {
                if (value.empty()) {
                    throw std::invalid_argument("--variant: empty name");
                }
                options.variants.emplace_back(value);
            } else if (name == "--threads") {
                options.threads.push_back(parse_size_arg(name, value));
            } else if (name == "--boundary") {
                options.boundary = parse_boundary_arg(name, value);
            } else if (name == "--output") {
                if (value.empty()) {
                    throw std::invalid_argument("--output: empty file name");
                }
                options.output = std::string(value);
            } else {
                std::ostringstream message;
                message << "unknown option: ";
                message << name;
                throw std::invalid_argument(message.str());
            }
        }

        // Check the settings the parser leaves to the library.
        void check(const BenchmarkOptions &options, std::span<const TMz::BackendType> backends,
                   std::span<const std::string> variants)
        {
            if (options.steps == 0) {
                throw std::invalid_argument("run_benchmark: steps must be positive");
            }
            if (options.repeat == 0) {
                throw std::invalid_argument("run_benchmark: repeat must be positive");
            }
            for (const auto type : backends) {
                if (!TMz::is_backend_available(type)) {
                    std::ostringstream message;
                    message << "run_benchmark: backend not compiled in: ";
                    message << TMz::backend_name(type);
                    throw std::invalid_argument(message.str());
                }
            }
            for (const std::string &variant : variants) {
                if (!backend_reads_variant(TMz::BackendType::OpenMP, variant) &&
                    !backend_reads_variant(TMz::BackendType::Cuda, variant)) {
                    std::ostringstream message;
                    message << "run_benchmark: no backend reads the variant ";
                    message << variant;
                    throw std::invalid_argument(message.str());
                }
            }
        }

        // Build the workload, time its run, and sum Ez once the fields are back.
        BenchmarkSample measure(const BenchmarkOptions &options, std::size_t ny,
                                TMz::BackendType backend, const std::string &variant,
                                std::size_t threads, std::size_t repeat)
        {
            TMz::Simulation::Options simOptions;
            simOptions.boundary = options.boundary;
            simOptions.backend = backend;
            parse_variant_arg("run_benchmark: variant", backend, variant, simOptions);
            simOptions.omp_threads = threads;
            TMz::Simulation sim(options.nx, ny, simOptions);
            sim.add_source(options.nx / 2, ny / 2, TMz::Component::Ez,
                           TMz::Gaussian::from_steps(sim.dt(), PULSE_WIDTH_STEPS));

            const auto start = std::chrono::steady_clock::now();
            sim.run(options.steps);
            const auto stop = std::chrono::steady_clock::now();

            sim.download();
            double checksum = 0;
            for (const FloatType value : sim.fields().ez()) {
                checksum += static_cast<double>(value);
            }
            const double seconds = std::chrono::duration<double>(stop - start).count();
            const TMz::StepTraffic traffic = sim.step_traffic();
            return {
                .backend = backend,
                .variant = variant_name(backend, simOptions),
                .threads = threads,
                .nx = options.nx,
                .ny = ny,
                .nsteps = options.steps,
                .repeat = repeat,
                .seconds = seconds,
                .checksum = checksum,
                .bytes_per_step = traffic.bytes,
                .flops_per_step = traffic.flops,
            };
        }

        // Seconds since a time point of the steady clock.
        double seconds_since(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }

        // Run the workload untimed until the warm-up time has passed.
        void warm_up(const BenchmarkOptions &options, std::size_t ny, TMz::BackendType backend,
                     const std::string &variant, std::size_t threads)
        {
            if (!(options.warmup > 0)) {
                return;
            }
            const auto start = std::chrono::steady_clock::now();
            do {
                measure(options, ny, backend, variant, threads, 0);
            } while (seconds_since(start) < options.warmup);
        }

        // Keep the names a backend reads, or default when it reads none of them.
        std::vector<std::string> variants_of(TMz::BackendType backend,
                                             std::span<const std::string> variants)
        {
            std::vector<std::string> names;
            for (const std::string &variant : variants) {
                if (backend_reads_variant(backend, variant)) {
                    names.push_back(variant);
                }
            }
            if (names.empty()) {
                names.emplace_back("default");
            }
            return names;
        }

        // Take the median of the sample times, the mean of the middle two for an even count.
        double median_seconds(std::span<const BenchmarkSample> samples)
        {
            std::vector<double> seconds;
            seconds.reserve(samples.size());
            for (const BenchmarkSample &sample : samples) {
                seconds.push_back(sample.seconds);
            }
            std::sort(seconds.begin(), seconds.end());
            const std::size_t mid = seconds.size() / 2;
            return seconds.size() % 2 == 1 ? seconds[mid] : (seconds[mid - 1] + seconds[mid]) / 2;
        }

        // Write the summary line of one configuration.
        void summarize(std::span<const BenchmarkSample> samples, std::ostream &log)
        {
            const BenchmarkSample &first = samples.front();
            const double median = median_seconds(samples);
            const double cells = static_cast<double>(first.nx) * static_cast<double>(first.ny) *
                                 static_cast<double>(first.nsteps);
            std::ostringstream line;
            line << TMz::backend_name(first.backend);
            line << ' ';
            line << first.variant;
            line << ", threads ";
            line << first.threads;
            line << ", ";
            line << first.nx;
            line << " x ";
            line << first.ny;
            line << ", ";
            line << first.nsteps;
            line << " steps: median ";
            line << std::fixed;
            line << std::setprecision(6);
            line << median;
            line << " s, ";
            line << std::setprecision(1);
            line << cells / median / 1e6;
            line << " Mcells/s, checksum ";
            line << std::defaultfloat;
            line << std::setprecision(17);
            line << first.checksum;
            line << '\n';
            log << line.str();
        }
    }

    // Split the arguments, then set the flags and apply each option with its value.
    BenchmarkOptions parse_benchmark_options(std::span<const std::string_view> args)
    {
        static constexpr std::string_view FLAGS[] = {
            "-h", "--help", "-v", "--version", "--header-only", "--no-header", "--peaks"};
        BenchmarkOptions options;
        for (const Argument &arg : split_arguments(args, FLAGS)) {
            if (arg.name == "-h" || arg.name == "--help") {
                options.help = true;
            } else if (arg.name == "-v" || arg.name == "--version") {
                options.version = true;
            } else if (arg.name == "--header-only") {
                options.header_only = true;
            } else if (arg.name == "--no-header") {
                options.no_header = true;
            } else if (arg.name == "--peaks") {
                options.peaks = true;
            } else {
                apply(options, arg.name, arg.value);
            }
        }
        return options;
    }

    // Build the usage text.
    std::string benchmark_usage(std::string_view program)
    {
        std::ostringstream usage;
        usage << "Usage: ";
        usage << program;
        usage << " [options]\n";
        usage << "\n";
        usage << "Measure the 2D TMz time loop: a vacuum grid with a Gaussian Ez source at the\n";
        usage << "centre, repeated per backend, variant and thread count, one CSV row per\n";
        usage << "sample. A summary per configuration goes to standard error.\n";
        usage << "\n";
        usage << "Options:\n";
        usage << "  --nx N                Cells along x (default: 256)\n";
        usage << "  --ny N                Cells along y (default: nx)\n";
        usage << "  --steps N             Time steps per sample (default: 500)\n";
        usage << "  --repeat N            Samples per configuration (default: 10)\n";
        usage << "  --warmup S            Seconds of untimed runs first, 0 for none\n";
        usage << "                        (default: 0.1)\n";
        usage << "  --order O             Sample order, grouped (each configuration's samples\n";
        usage << "                        in a row) or interleaved (one sample of each per\n";
        usage << "                        round) (default: grouped)\n";
        usage << "  --backend B           Backend, serial, openmp, cuda or all; repeatable\n";
        usage << "                        (default: all)\n";
        usage << "  --variant V           OpenMP variant SCHEDULE[:WxH][+simd][+phase], SCHEDULE\n";
        usage << "                        rows, collapse2, tiles or tasks; CUDA variant\n";
        usage << "                        KERNEL[:BXxBY][+table][+graph], KERNEL fused,\n";
        usage << "                        split or tiled; or default.\n";
        usage << "                        Repeatable; a backend runs the names it reads, or\n";
        usage << "                        default (default: default)\n";
        usage << "  --threads N           Thread count, for OpenMP only; repeatable\n";
        usage << "                        (default: 1)\n";
        usage << "  --boundary B          Outer boundary, pec, pmc or pml (default: pml)\n";
        usage << "  --output FILE         Write the CSV to FILE (default: standard output)\n";
        usage << "  --header-only         Write only the CSV header and exit\n";
        usage << "  --no-header           Leave the CSV header out\n";
        usage << "  --peaks               Write the machine's peak bandwidth and flops as\n";
        usage << "                        '# key: value' lines and exit\n";
        usage << "  -h, --help            Show this message and exit\n";
        usage << "  -v, --version         Show the version and exit\n";
        usage << "\n";
        usage << "Example:\n";
        usage << "  ";
        usage << program;
        usage << " --nx 1024 --steps 200 --repeat 5 --output bench.csv\n";
        return usage.str();
    }

    // List the columns.
    std::string benchmark_csv_header()
    {
        return "backend,variant,precision,threads,nx,ny,nsteps,repeat,seconds,checksum,"
               "bytes_per_step,flops_per_step";
    }

    // Format the columns in header order.
    std::string format_benchmark_row(const BenchmarkSample &sample)
    {
        std::ostringstream row;
        row << TMz::backend_name(sample.backend);
        row << ',';
        row << sample.variant;
        row << ',';
        row << PRECISION;
        row << ',';
        row << sample.threads;
        row << ',';
        row << sample.nx;
        row << ',';
        row << sample.ny;
        row << ',';
        row << sample.nsteps;
        row << ',';
        row << sample.repeat;
        row << ',';
        row << std::fixed;
        row << std::setprecision(9);
        row << sample.seconds;
        row << ',';
        row << std::defaultfloat;
        row << std::setprecision(17);
        row << sample.checksum;
        row << ',';
        row << std::fixed;
        row << std::setprecision(0);
        row << sample.bytes_per_step;
        row << ',';
        row << sample.flops_per_step;
        return row.str();
    }

    // Query and measure the GPU, then measure the CPU, each where it is built in.
    void write_benchmark_peaks(std::ostream &out)
    {
#if FDTD_WITH_CUDA
        if (TMz::is_backend_available(TMz::BackendType::Cuda)) {
            namespace gpu = TMz::backends::gpu;
            const gpu::DeviceInfo info = gpu::device_info();
            const auto [fp32, fp64] = lanes_per_sm(info.major, info.minor);
            const double sms = static_cast<double>(info.sms);
            const double bandwidth = 2 * info.memory_clock * static_cast<double>(info.bus_bits) / 8;
            peak_line(out, "gpu_sm_count", sms, 0, "");
            peak_line(out, "gpu_clock", info.clock / 1e6, 0, "MHz");
            peak_line(out, "gpu_l2_cache", static_cast<double>(info.l2_bytes) / (1 << 20), 1,
                      "MiB");
            peak_line(out, "gpu_peak_bandwidth", bandwidth / 1e9, 1, "GB/s");
            const double copy = gpu::copy_bandwidth(2 * PEAK_BYTES, PEAK_REPEATS);
            peak_line(out, "gpu_copy_bandwidth", copy / 1e9, 1, "GB/s");
            if (fp32 > 0) {
                peak_line(out, "gpu_peak_flops_float", 2 * fp32 * sms * info.clock / 1e9, 0,
                          "GFLOPS");
                peak_line(out, "gpu_peak_flops_double", 2 * fp64 * sms * info.clock / 1e9, 0,
                          "GFLOPS");
            }
        }
#endif
#if FDTD_WITH_OPENMP
        const auto best = TMz::backends::best_triad_bandwidth(PEAK_BYTES, PEAK_REPEATS);
        peak_line(out, "cpu_triad_bandwidth", best.bandwidth / 1e9, 1, "GB/s");
        peak_line(out, "cpu_triad_threads", static_cast<double>(best.threads), 0, "");
#endif
        out << std::flush;
    }

    // Resolve the defaults, then measure each configuration and write its rows.
    std::vector<BenchmarkSample> run_benchmark(const BenchmarkOptions &options, std::ostream &out,
                                               std::ostream &log)
    {
        const std::vector<TMz::BackendType> backends =
            options.backends.empty() ? TMz::available_backends() : options.backends;
        const std::vector<std::string> variants =
            options.variants.empty() ? std::vector<std::string>{"default"} : options.variants;
        const std::vector<std::size_t> threadCounts =
            options.threads.empty() ? std::vector<std::size_t>{1} : options.threads;
        const std::vector<std::size_t> oneThread{1};
        const std::size_t ny = options.ny > 0 ? options.ny : options.nx;
        if (!options.header_only) {
            check(options, backends, variants);
        }

        std::ofstream file;
        if (!options.output.empty()) {
            file.open(options.output);
            if (!file) {
                std::ostringstream message;
                message << "run_benchmark: cannot write ";
                message << options.output.string();
                throw std::runtime_error(message.str());
            }
        }
        std::ostream &csv = options.output.empty() ? out : file;
        if (options.peaks) {
            write_benchmark_peaks(csv);
            return {};
        }
        if (!options.no_header || options.header_only) {
            csv << benchmark_csv_header();
            csv << '\n';
        }

        std::vector<BenchmarkSample> samples;
        if (options.header_only) {
            return samples;
        }
        std::vector<Configuration> configurations;
        for (const auto backend : backends) {
            const std::vector<std::size_t> &counts =
                backend == TMz::BackendType::OpenMP ? threadCounts : oneThread;
            for (const std::string &variant : variants_of(backend, variants)) {
                for (const std::size_t threads : counts) {
                    configurations.push_back({backend, variant, threads});
                }
            }
        }

        auto sample = [&](const Configuration &c, std::size_t r) {
            samples.push_back(measure(options, ny, c.backend, c.variant, c.threads, r));
            csv << format_benchmark_row(samples.back());
            csv << '\n';
            csv << std::flush;
        };

        if (options.order == BenchmarkOrder::Grouped) {
            for (const Configuration &c : configurations) {
                warm_up(options, ny, c.backend, c.variant, c.threads);
                const std::size_t first = samples.size();
                for (std::size_t r = 0; r < options.repeat; ++r) {
                    sample(c, r);
                }
                summarize(std::span(samples).subspan(first), log);
            }
        } else {
            for (const Configuration &c : configurations) {
                warm_up(options, ny, c.backend, c.variant, c.threads);
            }
            for (std::size_t r = 0; r < options.repeat; ++r) {
                for (const Configuration &c : configurations) {
                    sample(c, r);
                }
            }
            const std::size_t count = configurations.size();
            for (std::size_t k = 0; k < count; ++k) {
                std::vector<BenchmarkSample> own;
                for (std::size_t r = 0; r < options.repeat; ++r) {
                    own.push_back(samples[r * count + k]);
                }
                summarize(own, log);
            }
        }
        if (!csv) {
            throw std::runtime_error("run_benchmark: failed writing the CSV");
        }
        return samples;
    }
}
