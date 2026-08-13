/*
 * C++ file for the fdtd command-line program.
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

// The driver and the version.
#include "fdtd/fdtd_driver.h"
#include "fdtd/fdtd_version.h"

// Standard library for streams and errors.
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

// Parse the arguments and run the driver, exiting 2 on bad usage and 1 on failure.
int main(int argc, char **argv)
{
    const std::vector<std::string_view> args(argv + 1, argv + argc);
    fdtd::DriverOptions options;
    try {
        options = fdtd::parse_driver_options(args);
    } catch (const std::exception &error) {
        std::cerr << "fdtd: ";
        std::cerr << error.what();
        std::cerr << "\nTry 'fdtd --help'.\n";
        return 2;
    }
    if (options.help) {
        std::cout << fdtd::driver_usage("fdtd");
        return 0;
    }
    if (options.version) {
        std::cout << FDTD_VERSION_STRING;
        std::cout << '\n';
        return 0;
    }
    try {
        fdtd::run_driver(options, std::cout);
    } catch (const std::exception &error) {
        std::cerr << "fdtd: ";
        std::cerr << error.what();
        std::cerr << '\n';
        return 1;
    }
    return 0;
}
