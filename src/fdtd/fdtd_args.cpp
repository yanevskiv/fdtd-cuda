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

// Standard library for parsing, math, string building, errors and pairs.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

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

        // Throw for a malformed variant name with the reason.
        [[noreturn]] void malformed_variant(std::string_view name, std::string_view text,
                                            std::string_view reason)
        {
            std::ostringstream message;
            message << name;
            message << ": ";
            message << reason;
            message << ": ";
            message << text;
            throw std::invalid_argument(message.str());
        }

        // Name an OpenMP schedule.
        std::string_view schedule_name(TMz::OmpSchedule schedule) noexcept
        {
            switch (schedule) {
                case TMz::OmpSchedule::Rows:
                    return "rows";
                case TMz::OmpSchedule::Collapse2:
                    return "collapse2";
                case TMz::OmpSchedule::Tiles:
                    return "tiles";
                case TMz::OmpSchedule::Tasks:
                    break;
            }
            return "tasks";
        }

        // Name a CUDA kernel layout.
        std::string_view kernel_name(TMz::CudaKernel kernel) noexcept
        {
            switch (kernel) {
                case TMz::CudaKernel::Fused:
                    return "fused";
                case TMz::CudaKernel::Split:
                    break;
            }
            return "split";
        }

        // Read a positive decimal side, the whole field, of a tile or a block.
        std::size_t parse_side(std::string_view name, std::string_view text, std::string_view side,
                               std::string_view what)
        {
            std::size_t value = 0;
            const auto [end, error] =
                std::from_chars(side.data(), side.data() + side.size(), value);
            if (side.empty() || error != std::errc() || end != side.data() + side.size() ||
                value == 0) {
                std::ostringstream reason;
                reason << what;
                reason << " sides must be positive integers";
                malformed_variant(name, text, reason.str());
            }
            return value;
        }

        // Read the two sides of a tile or a block written as WxH.
        std::pair<std::size_t, std::size_t> parse_sides(std::string_view name,
                                                        std::string_view text,
                                                        std::string_view size,
                                                        std::string_view what)
        {
            const std::size_t cross = size.find('x');
            if (cross == std::string_view::npos) {
                std::ostringstream reason;
                reason << "expected a ";
                reason << what;
                reason << " size as WxH";
                malformed_variant(name, text, reason.str());
            }
            return {
                parse_side(name, text, size.substr(0, cross), what),
                parse_side(name, text, size.substr(cross + 1), what),
            };
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
    TMz::BackendType parse_backend_arg(std::string_view name, std::string_view text)
    {
        for (const auto type :
             {TMz::BackendType::Serial, TMz::BackendType::OpenMP, TMz::BackendType::Cuda}) {
            if (text == TMz::backend_name(type)) {
                return type;
            }
        }
        std::ostringstream message;
        message << name;
        message << ": expected serial, openmp or cuda: ";
        message << text;
        throw std::invalid_argument(message.str());
    }

    // Split off the suffixes, then the tile size, and read the schedule into a copy.
    void parse_omp_variant_arg(std::string_view name, std::string_view text,
                               TMz::Simulation::Options &options)
    {
        const TMz::Simulation::Options defaults;
        TMz::Simulation::Options parsed = options;
        parsed.omp_schedule = defaults.omp_schedule;
        parsed.omp_tile_x = defaults.omp_tile_x;
        parsed.omp_tile_y = defaults.omp_tile_y;
        parsed.omp_simd = defaults.omp_simd;
        parsed.omp_region = defaults.omp_region;
        if (text == "default") {
            options = parsed;
            return;
        }
        const std::string_view head = text.substr(0, text.find('+'));
        std::string_view suffixes = text.substr(head.size());
        bool simd = false;
        bool phase = false;
        while (!suffixes.empty()) {
            suffixes.remove_prefix(1);
            const std::string_view suffix = suffixes.substr(0, suffixes.find('+'));
            suffixes.remove_prefix(suffix.size());
            if (suffix == "simd" && !simd) {
                simd = true;
            } else if (suffix == "phase" && !phase) {
                phase = true;
            } else {
                malformed_variant(name, text, "expected +simd or +phase, each once");
            }
        }
        parsed.omp_simd = simd;
        parsed.omp_region = phase ? TMz::OmpRegion::Phase : TMz::OmpRegion::Run;

        const std::size_t colon = head.find(':');
        const std::string_view schedule = head.substr(0, colon);
        if (schedule == "rows") {
            parsed.omp_schedule = TMz::OmpSchedule::Rows;
        } else if (schedule == "collapse2") {
            parsed.omp_schedule = TMz::OmpSchedule::Collapse2;
        } else if (schedule == "tiles") {
            parsed.omp_schedule = TMz::OmpSchedule::Tiles;
        } else if (schedule == "tasks") {
            parsed.omp_schedule = TMz::OmpSchedule::Tasks;
        } else {
            malformed_variant(name, text, "expected rows, collapse2, tiles or tasks");
        }
        if (colon != std::string_view::npos) {
            if (parsed.omp_schedule != TMz::OmpSchedule::Tiles &&
                parsed.omp_schedule != TMz::OmpSchedule::Tasks) {
                malformed_variant(name, text, "a tile size is for tiles and tasks only");
            }
            const auto [x, y] = parse_sides(name, text, head.substr(colon + 1), "tile");
            parsed.omp_tile_x = x;
            parsed.omp_tile_y = y;
        }
        options = parsed;
    }

    // Write the schedule, the tile size where it counts, then the suffixes.
    std::string omp_variant_name(const TMz::Simulation::Options &options)
    {
        std::ostringstream name;
        name << schedule_name(options.omp_schedule);
        if (options.omp_schedule == TMz::OmpSchedule::Tiles ||
            options.omp_schedule == TMz::OmpSchedule::Tasks) {
            name << ':';
            name << options.omp_tile_x;
            name << 'x';
            name << options.omp_tile_y;
        }
        if (options.omp_simd) {
            name << "+simd";
        }
        if (options.omp_region == TMz::OmpRegion::Phase) {
            name << "+phase";
        }
        return name.str();
    }

    // Read the kernel, then the block when there is one.
    void parse_cuda_variant_arg(std::string_view name, std::string_view text,
                                TMz::Simulation::Options &options)
    {
        const TMz::Simulation::Options defaults;
        TMz::Simulation::Options parsed = options;
        parsed.cuda_kernel = defaults.cuda_kernel;
        parsed.cuda_block_x = defaults.cuda_block_x;
        parsed.cuda_block_y = defaults.cuda_block_y;
        parsed.cuda_graph = defaults.cuda_graph;
        if (text == "default") {
            options = parsed;
            return;
        }
        const std::string_view head = text.substr(0, text.find('+'));
        std::string_view suffixes = text.substr(head.size());
        bool graph = false;
        while (!suffixes.empty()) {
            suffixes.remove_prefix(1);
            const std::string_view suffix = suffixes.substr(0, suffixes.find('+'));
            suffixes.remove_prefix(suffix.size());
            if (suffix == "graph" && !graph) {
                graph = true;
            } else {
                malformed_variant(name, text, "expected +graph, once");
            }
        }
        parsed.cuda_graph = graph;

        const std::size_t colon = head.find(':');
        const std::string_view kernel = head.substr(0, colon);
        if (kernel == "fused") {
            parsed.cuda_kernel = TMz::CudaKernel::Fused;
        } else if (kernel == "split") {
            parsed.cuda_kernel = TMz::CudaKernel::Split;
        } else {
            malformed_variant(name, text, "expected fused or split");
        }
        if (colon != std::string_view::npos) {
            const auto [x, y] = parse_sides(name, text, head.substr(colon + 1), "block");
            parsed.cuda_block_x = x;
            parsed.cuda_block_y = y;
        }
        options = parsed;
    }

    // Write the kernel and the block, then the suffixes.
    std::string cuda_variant_name(const TMz::Simulation::Options &options)
    {
        std::ostringstream name;
        name << kernel_name(options.cuda_kernel);
        name << ':';
        name << options.cuda_block_x;
        name << 'x';
        name << options.cuda_block_y;
        if (options.cuda_graph) {
            name << "+graph";
        }
        return name.str();
    }

    // Try the backend's grammar on the name.
    bool backend_reads_variant(TMz::BackendType backend, std::string_view text)
    {
        TMz::Simulation::Options scratch;
        try {
            parse_variant_arg("variant", backend, text, scratch);
        } catch (const std::invalid_argument &) {
            return false;
        }
        return true;
    }

    // Hand the name to the backend's grammar; serial reads default only.
    void parse_variant_arg(std::string_view name, TMz::BackendType backend, std::string_view text,
                           TMz::Simulation::Options &options)
    {
        switch (backend) {
            case TMz::BackendType::Serial:
                if (text != "default") {
                    malformed_variant(name, text, "the serial backend has no variants");
                }
                return;
            case TMz::BackendType::OpenMP:
                parse_omp_variant_arg(name, text, options);
                return;
            case TMz::BackendType::Cuda:
                break;
        }
        parse_cuda_variant_arg(name, text, options);
    }

    // Name the variant in the backend's grammar.
    std::string variant_name(TMz::BackendType backend, const TMz::Simulation::Options &options)
    {
        switch (backend) {
            case TMz::BackendType::Serial:
                return "default";
            case TMz::BackendType::OpenMP:
                return omp_variant_name(options);
            case TMz::BackendType::Cuda:
                break;
        }
        return cuda_variant_name(options);
    }
}
