/*
 * C++ file for command-line argument parsing shared by the programs.
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
#include "fdtd/fdtd_args.h"

// Standard library for parsing, math, string building and errors.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

// Namespace for the fdtd library.
namespace fdtd
{
    // Helpers private to this file.
    namespace
    {
        // Parse a whole finite number, or return NaN when the text is not one.
        double parse_number(std::string_view text)
        {
            double value = 0;
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc() || end != text.data() + text.size() || !std::isfinite(value)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return value;
        }
    }

    // Walk the arguments, taking each option's value after '=' or from the next argument.
    std::vector<Argument> split_arguments(std::span<const std::string_view> args,
                                          std::span<const std::string_view> flags)
    {
        std::vector<Argument> options;
        for (std::size_t k = 0; k < args.size(); ++k) {
            const std::string_view arg = args[k];
            if (std::find(flags.begin(), flags.end(), arg) != flags.end()) {
                options.push_back({arg, {}});
                continue;
            }
            if (!arg.starts_with("--")) {
                std::ostringstream message;
                message << "unexpected argument: ";
                message << arg;
                throw std::invalid_argument(message.str());
            }
            const std::size_t equals = arg.find('=');
            if (equals != std::string_view::npos) {
                const std::string_view name = arg.substr(0, equals);
                if (std::find(flags.begin(), flags.end(), name) != flags.end()) {
                    std::ostringstream message;
                    message << name;
                    message << ": takes no value";
                    throw std::invalid_argument(message.str());
                }
                options.push_back({name, arg.substr(equals + 1)});
                continue;
            }
            if (k + 1 == args.size()) {
                std::ostringstream message;
                message << arg;
                message << ": missing value";
                throw std::invalid_argument(message.str());
            }
            options.push_back({arg, args[++k]});
        }
        return options;
    }

    // Parse with from_chars and require the whole text to be used.
    std::size_t parse_size_arg(std::string_view name, std::string_view text)
    {
        std::size_t value = 0;
        const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (ec != std::errc() || end != text.data() + text.size()) {
            std::ostringstream message;
            message << name;
            message << ": not a whole number: ";
            message << text;
            throw std::invalid_argument(message.str());
        }
        return value;
    }

    // Parse a finite number and require it above zero.
    double parse_positive_arg(std::string_view name, std::string_view text)
    {
        const double value = parse_number(text);
        if (!(value > 0)) {
            std::ostringstream message;
            message << name;
            message << ": not a positive number: ";
            message << text;
            throw std::invalid_argument(message.str());
        }
        return value;
    }

    // Parse a finite number and require it at or above zero.
    double parse_nonnegative_arg(std::string_view name, std::string_view text)
    {
        const double value = parse_number(text);
        if (!(value >= 0)) {
            std::ostringstream message;
            message << name;
            message << ": not a non-negative number: ";
            message << text;
            throw std::invalid_argument(message.str());
        }
        return value;
    }

    // Cut the text at every comma.
    std::vector<std::string_view> split_list_arg(std::string_view text)
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

    // Match the three boundary names.
    TMz::Boundary parse_boundary_arg(std::string_view name, std::string_view text)
    {
        if (text == "pec") {
            return TMz::Boundary::Pec;
        }
        if (text == "pmc") {
            return TMz::Boundary::Pmc;
        }
        if (text == "pml") {
            return TMz::Boundary::Pml;
        }
        std::ostringstream message;
        message << name;
        message << ": expected pec, pmc or pml: ";
        message << text;
        throw std::invalid_argument(message.str());
    }

    // Match the name of every backend type.
    TMz::backends::BackendType parse_backend_arg(std::string_view name, std::string_view text)
    {
        for (const auto type :
             {TMz::backends::BackendType::Serial, TMz::backends::BackendType::OpenMP,
              TMz::backends::BackendType::Cuda}) {
            if (text == TMz::backends::backend_name(type)) {
                return type;
            }
        }
        std::ostringstream message;
        message << name;
        message << ": expected serial, openmp or cuda: ";
        message << text;
        throw std::invalid_argument(message.str());
    }
}
