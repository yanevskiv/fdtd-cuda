/*
 * C++ file for the options of the 2D TMz OpenMP backend.
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
#include "fdtd/TMz/fdtd_tmz_omp_options.h"

// Standard library for number parsing, string building and errors.
#include <charconv>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Helpers private to this file.
    namespace
    {
        // Throw for a malformed name with the reason.
        [[noreturn]] void malformed(std::string_view name, std::string_view reason)
        {
            std::ostringstream message;
            message << "parse_omp_variant: ";
            message << reason;
            message << ": ";
            message << name;
            throw std::invalid_argument(message.str());
        }

        // Name a schedule.
        std::string_view schedule_name(OmpSchedule schedule) noexcept
        {
            switch (schedule) {
                case OmpSchedule::Rows:
                    return "rows";
                case OmpSchedule::Collapse2:
                    return "collapse2";
                case OmpSchedule::Tiles:
                    return "tiles";
                case OmpSchedule::Tasks:
                    break;
            }
            return "tasks";
        }

        // Read a positive decimal tile side, the whole text.
        std::size_t parse_side(std::string_view name, std::string_view text)
        {
            std::size_t value = 0;
            const auto [end, error] =
                std::from_chars(text.data(), text.data() + text.size(), value);
            if (text.empty() || error != std::errc() || end != text.data() + text.size() ||
                value == 0) {
                malformed(name, "tile sides must be positive integers");
            }
            return value;
        }
    }

    // Split off the suffixes, then the tile size, and read the schedule.
    OmpOptions parse_omp_variant(std::string_view name)
    {
        OmpOptions options;
        if (name == "default") {
            return options;
        }
        std::string_view head = name.substr(0, name.find('+'));
        std::string_view suffixes = name.substr(head.size());
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
                malformed(name, "expected +simd or +phase, each once");
            }
        }
        options.simd = simd;
        options.region = phase ? OmpRegion::Phase : OmpRegion::Run;

        const std::size_t colon = head.find(':');
        const std::string_view schedule = head.substr(0, colon);
        if (schedule == "rows") {
            options.schedule = OmpSchedule::Rows;
        } else if (schedule == "collapse2") {
            options.schedule = OmpSchedule::Collapse2;
        } else if (schedule == "tiles") {
            options.schedule = OmpSchedule::Tiles;
        } else if (schedule == "tasks") {
            options.schedule = OmpSchedule::Tasks;
        } else {
            malformed(name, "expected rows, collapse2, tiles or tasks");
        }
        if (colon == std::string_view::npos) {
            return options;
        }
        if (options.schedule != OmpSchedule::Tiles && options.schedule != OmpSchedule::Tasks) {
            malformed(name, "a tile size is for tiles and tasks only");
        }
        const std::string_view size = head.substr(colon + 1);
        const std::size_t cross = size.find('x');
        if (cross == std::string_view::npos) {
            malformed(name, "expected a tile size as WxH");
        }
        options.tile_x = parse_side(name, size.substr(0, cross));
        options.tile_y = parse_side(name, size.substr(cross + 1));
        return options;
    }

    // Write the schedule, the tile size where it counts, then the suffixes.
    std::string omp_variant_name(const OmpOptions &options)
    {
        std::ostringstream name;
        name << schedule_name(options.schedule);
        if (options.schedule == OmpSchedule::Tiles || options.schedule == OmpSchedule::Tasks) {
            name << ':';
            name << options.tile_x;
            name << 'x';
            name << options.tile_y;
        }
        if (options.simd) {
            name << "+simd";
        }
        if (options.region == OmpRegion::Phase) {
            name << "+phase";
        }
        return name.str();
    }
}
