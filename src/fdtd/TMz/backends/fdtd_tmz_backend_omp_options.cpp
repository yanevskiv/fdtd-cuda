/*
 * C++ file for the settings of the 2D TMz OpenMP backend and the variant names they read from.
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

// Class declarations.
#include "fdtd/TMz/backends/fdtd_tmz_backend_omp_options.h"

// Standard library for sizes, string building, errors and pairs.
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Helpers private to this file.
    namespace
    {
        // Throw for a malformed variant name with the reason.
        [[noreturn]] void malformed(std::string_view variant, std::string_view reason)
        {
            std::ostringstream message;
            message << reason;
            message << ": ";
            message << variant;
            throw std::invalid_argument(message.str());
        }

        // The levels in order, with their names.
        constexpr std::pair<OmpLevel, std::string_view> LEVELS[] = {
            {OmpLevel::V1, "v1"},
            {OmpLevel::V2, "v2"},
            {OmpLevel::V3, "v3"},
            {OmpLevel::V4, "v4"},
        };
    }

    // Read the level, then the suffix.
    OmpOptions OmpOptions::parse(std::string_view variant)
    {
        OmpOptions parsed;
        if (variant == "default") {
            return parsed;
        }
        const std::size_t plus = variant.find('+');
        const std::string_view head = variant.substr(0, plus);
        bool known = false;
        for (const auto &[level, name] : LEVELS) {
            if (head == name) {
                parsed.level = level;
                known = true;
            }
        }
        if (!known) {
            malformed(variant, "expected v1, v2, v3 or v4");
        }
        if (plus == std::string_view::npos) {
            return parsed;
        }
        const std::string_view suffix = variant.substr(plus + 1);
        if (suffix == "close") {
            parsed.bind = OmpBind::Close;
        } else if (suffix == "spread") {
            parsed.bind = OmpBind::Spread;
        } else {
            malformed(variant, "expected +close or +spread, once");
        }
        return parsed;
    }

    // Write the level, then the suffix.
    std::string OmpOptions::name() const
    {
        std::ostringstream text;
        for (const auto &[value, name] : LEVELS) {
            if (value == level) {
                text << name;
            }
        }
        if (bind == OmpBind::Close) {
            text << "+close";
        } else if (bind == OmpBind::Spread) {
            text << "+spread";
        }
        return text.str();
    }
}
