/*
 * C++ file for the benchmark program tests.
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

#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"
#include "fdtd/fdtd_benchmark.h"
#include "fdtd/fdtd_float.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using fdtd::BenchmarkOptions;
using fdtd::BenchmarkSample;
using fdtd::TMz::BackendType;
using fdtd::TMz::Boundary;

namespace
{
    // Parse a list of arguments.
    BenchmarkOptions parse(std::vector<std::string_view> args)
    {
        return fdtd::parse_benchmark_options(args);
    }

    // Small, non-square settings that run fast with the PML on: 2 samples, no warm-up.
    BenchmarkOptions small()
    {
        BenchmarkOptions options;
        options.nx = 48;
        options.ny = 44;
        options.steps = 20;
        options.repeat = 2;
        options.warmup = 0;
        return options;
    }

    // Run the benchmark into a string, discarding the log.
    std::vector<BenchmarkSample> run(const BenchmarkOptions &options, std::string &csv)
    {
        std::ostringstream out;
        std::ostringstream log;
        auto samples = fdtd::run_benchmark(options, out, log);
        csv = out.str();
        return samples;
    }

    // Split text into lines, without the newlines.
    std::vector<std::string> lines(const std::string &text)
    {
        std::vector<std::string> result;
        std::istringstream in(text);
        for (std::string line; std::getline(in, line);) {
            result.push_back(line);
        }
        return result;
    }

    // Run the benchmark workload by hand with Simulation and sum Ez.
    double workload_checksum(std::size_t nx, std::size_t ny, std::size_t steps, Boundary boundary)
    {
        fdtd::TMz::Simulation::Options options;
        options.boundary = boundary;
        fdtd::TMz::Simulation sim(nx, ny, options);
        sim.add_source(nx / 2, ny / 2, fdtd::TMz::Component::Ez,
                       fdtd::TMz::Gaussian::from_steps(sim.dt(), 12));
        sim.run(steps);
        sim.download();
        double sum = 0;
        for (const fdtd::FloatType value : sim.fields().ez()) {
            sum += static_cast<double>(value);
        }
        return sum;
    }
}

// Verify the defaults when no option is given.
TEST_CASE("Benchmark defaults", "[unit][benchmark]")
{
    const auto options = parse({});
    CHECK(options.nx == 256);
    CHECK(options.ny == 0);
    CHECK(options.steps == 500);
    CHECK(options.repeat == 10);
    CHECK(options.warmup == 0.1);
    CHECK(options.backends.empty());
    CHECK(options.variants.empty());
    CHECK(options.threads.empty());
    CHECK(options.boundary == Boundary::Pml);
    CHECK(options.output.empty());
    CHECK_FALSE(options.header_only);
    CHECK_FALSE(options.no_header);
    CHECK_FALSE(options.help);
    CHECK_FALSE(options.version);
}

// Verify every option, both value forms, and the repeatable ones in order.
TEST_CASE("Benchmark parses every option", "[unit][benchmark]")
{
    const auto options = parse({"--nx",
                                "64",
                                "--ny=32",
                                "--steps",
                                "40",
                                "--repeat=3",
                                "--warmup",
                                "0",
                                "--backend",
                                "serial",
                                "--backend=cuda",
                                "--variant",
                                "default",
                                "--variant=fast",
                                "--threads",
                                "2",
                                "--threads=8",
                                "--boundary",
                                "pec",
                                "--output",
                                "bench.csv",
                                "--header-only",
                                "--no-header",
                                "-h",
                                "--version"});
    CHECK(options.nx == 64);
    CHECK(options.ny == 32);
    CHECK(options.steps == 40);
    CHECK(options.repeat == 3);
    CHECK(options.warmup == 0);
    CHECK(options.backends == std::vector<BackendType>{BackendType::Serial, BackendType::Cuda});
    CHECK(options.variants == std::vector<std::string>{"default", "fast"});
    CHECK(options.threads == std::vector<std::size_t>{2, 8});
    CHECK(options.boundary == Boundary::Pec);
    CHECK(options.output == "bench.csv");
    CHECK(options.header_only);
    CHECK(options.no_header);
    CHECK(options.help);
    CHECK(options.version);
}

// Verify that "all" adds every compiled backend once, after any given before it.
TEST_CASE("Benchmark backend all lists the registry", "[unit][benchmark]")
{
    CHECK(parse({"--backend", "all"}).backends == fdtd::TMz::available_backends());
    CHECK(parse({"--backend", "all", "--backend", "serial"}).backends ==
          fdtd::TMz::available_backends());
    CHECK(parse({"--backend", "serial", "--backend", "all"}).backends.front() ==
          BackendType::Serial);
}

// Verify that malformed arguments throw.
TEST_CASE("Benchmark rejects bad arguments", "[unit][benchmark]")
{
    CHECK_THROWS_AS(parse({"--nx"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--nx", "ten"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--steps=-1"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--repeat", "1.5"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--warmup", "-0.1"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--warmup", "nan"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--backend", "gpu"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--variant="}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--threads", "x"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--boundary", "abc"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--output="}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--header-only=yes"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"--frobnicate", "1"}), std::invalid_argument);
    CHECK_THROWS_AS(parse({"stray"}), std::invalid_argument);
}

// Verify the header and the format of a row, the checksum with all 17 digits.
TEST_CASE("Benchmark CSV header and row format", "[unit][benchmark]")
{
    CHECK(fdtd::benchmark_csv_header() ==
          "backend,variant,precision,threads,nx,ny,nsteps,repeat,seconds,checksum");
    const BenchmarkSample sample{BackendType::Serial, "default", 1, 64, 32, 100, 3, 0.25, 0.1};
    const std::string expected =
        FDTD_DOUBLE ? "serial,default,double,1,64,32,100,3,0.250000000,0.10000000000000001"
                    : "serial,default,float,1,64,32,100,3,0.250000000,0.10000000000000001";
    CHECK(fdtd::format_benchmark_row(sample) == expected);
}

// Verify one row per sample, matching the samples returned and sharing one checksum.
TEST_CASE("Benchmark writes a row per sample", "[unit][benchmark]")
{
    auto options = small();
    options.repeat = 3;
    options.backends = {BackendType::Serial};
    std::string csv;
    const auto samples = run(options, csv);
    REQUIRE(samples.size() == 3);
    const auto rows = lines(csv);
    REQUIRE(rows.size() == 4);
    CHECK(rows[0] == fdtd::benchmark_csv_header());
    for (std::size_t r = 0; r < samples.size(); ++r) {
        CHECK(rows[r + 1] == fdtd::format_benchmark_row(samples[r]));
        CHECK(samples[r].backend == BackendType::Serial);
        CHECK(samples[r].variant == "default");
        CHECK(samples[r].threads == 1);
        CHECK(samples[r].nx == 48);
        CHECK(samples[r].ny == 44);
        CHECK(samples[r].nsteps == 20);
        CHECK(samples[r].repeat == r);
        CHECK(samples[r].seconds > 0);
        CHECK(samples[r].checksum == samples[0].checksum);
    }
}

// Verify that the checksum equals the sum of Ez of the workload run by hand.
TEST_CASE("Benchmark checksum matches Simulation run by hand", "[unit][benchmark]")
{
    auto options = small();
    options.repeat = 1;
    std::string csv;
    const double pml = run(options, csv).front().checksum;
    CHECK(pml == workload_checksum(48, 44, 20, Boundary::Pml));
    options.boundary = Boundary::Pec;
    const double pec = run(options, csv).front().checksum;
    CHECK(pec == workload_checksum(48, 44, 20, Boundary::Pec));
    CHECK(pml != pec);
    CHECK(pml != 0);
}

// Verify that warming up leaves the measured samples unchanged.
TEST_CASE("Benchmark warm-up does not leak into samples", "[unit][benchmark]")
{
    auto options = small();
    std::string csv;
    const double cold = run(options, csv).front().checksum;
    options.warmup = 0.01;
    for (const BenchmarkSample &sample : run(options, csv)) {
        CHECK(sample.checksum == cold);
    }
}

// Verify that ny defaults to nx and that serial runs once whatever the thread counts.
TEST_CASE("Benchmark resolves ny and serial threads", "[unit][benchmark]")
{
    auto options = small();
    options.ny = 0;
    options.repeat = 1;
    options.backends = {BackendType::Serial};
    options.threads = {2, 4, 8};
    std::string csv;
    const auto samples = run(options, csv);
    REQUIRE(samples.size() == 1);
    CHECK(samples[0].ny == 48);
    CHECK(samples[0].threads == 1);
}

// Verify --header-only and --no-header.
TEST_CASE("Benchmark header flags", "[unit][benchmark]")
{
    auto options = small();
    options.header_only = true;
    std::string csv;
    CHECK(run(options, csv).empty());
    CHECK(lines(csv) == std::vector<std::string>{fdtd::benchmark_csv_header()});
    CHECK(csv.ends_with('\n'));

    options.header_only = false;
    options.no_header = true;
    options.repeat = 1;
    const auto samples = run(options, csv);
    REQUIRE(samples.size() == fdtd::TMz::available_backends().size());
    CHECK(lines(csv).size() == samples.size());
    CHECK(lines(csv).front() == fdtd::format_benchmark_row(samples.front()));
}

// Verify that the library rejects the settings the parser lets through.
TEST_CASE("Benchmark rejects bad settings", "[unit][benchmark]")
{
    std::string csv;
    auto options = small();
    options.steps = 0;
    CHECK_THROWS_AS(run(options, csv), std::invalid_argument);
    options = small();
    options.repeat = 0;
    CHECK_THROWS_AS(run(options, csv), std::invalid_argument);
    options = small();
    options.variants = {"fast"};
    CHECK_THROWS_AS(run(options, csv), std::invalid_argument);
    options = small();
    options.nx = 0;
    CHECK_THROWS_AS(run(options, csv), std::invalid_argument);
    options = small();
    options.nx = 30;
    CHECK_THROWS_AS(run(options, csv), std::invalid_argument);
    for (const auto type : {BackendType::OpenMP, BackendType::Cuda}) {
        if (!fdtd::TMz::is_backend_available(type)) {
            options = small();
            options.backends = {type};
            CHECK_THROWS_AS(run(options, csv), std::invalid_argument);
        }
    }
}

// Verify that the CSV goes to the output file and the summary to the log.
TEST_CASE("Benchmark writes the output file and the summary", "[unit][benchmark]")
{
    const auto path = std::filesystem::temp_directory_path() / "fdtd_test_benchmark.csv";
    std::filesystem::remove(path);
    auto options = small();
    options.backends = {BackendType::Serial};
    options.output = path;
    std::ostringstream out;
    std::ostringstream log;
    const auto samples = fdtd::run_benchmark(options, out, log);
    CHECK(out.str().empty());
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    const auto rows = lines(text.str());
    REQUIRE(rows.size() == 3);
    CHECK(rows[0] == fdtd::benchmark_csv_header());
    CHECK(rows[2] == fdtd::format_benchmark_row(samples[1]));
    const auto summary = lines(log.str());
    REQUIRE(summary.size() == 1);
    CHECK(summary[0].starts_with("serial default, threads 1, 48 x 44, 20 steps: median "));
    CHECK(summary[0].find(" Mcells/s, checksum ") != std::string::npos);
    std::filesystem::remove(path);

    options.output = std::filesystem::temp_directory_path() / "no_such_dir" / "bench.csv";
    CHECK_THROWS_AS(fdtd::run_benchmark(options, out, log), std::runtime_error);
}
