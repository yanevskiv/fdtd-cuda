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

// Standard library for string building and errors.
#include <sstream>
#include <stdexcept>
#include <string>

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

        // Name a schedule.
        std::string_view schedule_name(OmpSchedule schedule) noexcept
        {
            switch (schedule) {
                case OmpSchedule::Rows:
                    return "rows";
                case OmpSchedule::Collapse2:
                    break;
            }
            return "collapse2";
        }
    }

    // Split off the suffixes, then read the schedule.
    OmpOptions OmpOptions::parse(std::string_view variant)
    {
        OmpOptions parsed;
        if (variant == "default") {
            return parsed;
        }
        const std::string_view head = variant.substr(0, variant.find('+'));
        std::string_view suffixes = variant.substr(head.size());
        bool simd = false;
        bool table = false;
        bool fuse = false;
        bool phase = false;
        OmpBind bind = OmpBind::Default;
        while (!suffixes.empty()) {
            suffixes.remove_prefix(1);
            const std::string_view suffix = suffixes.substr(0, suffixes.find('+'));
            suffixes.remove_prefix(suffix.size());
            if (suffix == "simd" && !simd) {
                simd = true;
            } else if (suffix == "table" && !table) {
                table = true;
            } else if (suffix == "fuse" && !fuse) {
                fuse = true;
            } else if (suffix == "phase" && !phase) {
                phase = true;
            } else if (suffix == "close" && bind == OmpBind::Default) {
                bind = OmpBind::Close;
            } else if (suffix == "spread" && bind == OmpBind::Default) {
                bind = OmpBind::Spread;
            } else {
                malformed(variant,
                          "expected +simd, +table, +fuse, +phase and one of +close or +spread");
            }
        }
        parsed.bind = bind;
        parsed.simd = simd;
        parsed.table = table;
        parsed.fuse = fuse;
        parsed.region = phase ? OmpRegion::Phase : OmpRegion::Run;

        if (head == "rows") {
            parsed.schedule = OmpSchedule::Rows;
        } else if (head == "collapse2") {
            parsed.schedule = OmpSchedule::Collapse2;
        } else {
            malformed(variant, "expected rows or collapse2");
        }
        return parsed;
    }

    // Write the schedule, then the suffixes.
    std::string OmpOptions::name() const
    {
        std::ostringstream text;
        text << schedule_name(schedule);
        if (simd) {
            text << "+simd";
        }
        if (table) {
            text << "+table";
        }
        if (fuse) {
            text << "+fuse";
        }
        if (region == OmpRegion::Phase) {
            text << "+phase";
        }
        if (bind == OmpBind::Close) {
            text << "+close";
        } else if (bind == OmpBind::Spread) {
            text << "+spread";
        }
        return text.str();
    }
}
