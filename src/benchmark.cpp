/*
 * C++ file for the benchmark program.
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

// The benchmark and the version.
#include "fdtd/fdtd_benchmark.h"
#include "fdtd/fdtd_version.h"

// Standard library for streams and errors.
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

// Parse the arguments and run the benchmark, exiting 2 on bad usage and 1 on failure.
int main(int argc, char **argv)
{
    const std::vector<std::string_view> args(argv + 1, argv + argc);
    fdtd::BenchmarkOptions options;
    try {
        options = fdtd::parse_benchmark_options(args);
    } catch (const std::exception &error) {
        std::cerr << "benchmark: ";
        std::cerr << error.what();
        std::cerr << "\nTry 'benchmark --help'.\n";
        return 2;
    }
    if (options.help) {
        std::cout << fdtd::benchmark_usage("benchmark");
        return 0;
    }
    if (options.version) {
        std::cout << FDTD_VERSION_STRING;
        std::cout << '\n';
        return 0;
    }
    try {
        fdtd::run_benchmark(options, std::cout, std::cerr);
    } catch (const std::exception &error) {
        std::cerr << "benchmark: ";
        std::cerr << error.what();
        std::cerr << '\n';
        return 1;
    }
    return 0;
}
