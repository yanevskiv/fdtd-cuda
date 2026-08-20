/*
 * C++ file for the settings of the 2D TMz CUDA backend and the variant names they read from.
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
#include "fdtd/TMz/backends/fdtd_tmz_backend_cuda_options.h"

// Standard library for parsing, string building and errors.
#include <charconv>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

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

        // Name a kernel layout.
        std::string_view kernel_name(CudaKernel kernel) noexcept
        {
            switch (kernel) {
                case CudaKernel::Fused:
                    return "fused";
                case CudaKernel::Tiled:
                    break;
            }
            return "tiled";
        }

        // Read a positive decimal side of the block, the whole field.
        std::size_t parse_side(std::string_view variant, std::string_view side)
        {
            std::size_t value = 0;
            const auto [end, error] =
                std::from_chars(side.data(), side.data() + side.size(), value);
            if (side.empty() || error != std::errc() || end != side.data() + side.size() ||
                value == 0) {
                malformed(variant, "block sides must be positive integers");
            }
            return value;
        }
    }

    // Split off the suffixes, then read the kernel and the block when there is one.
    CudaOptions CudaOptions::parse(std::string_view variant)
    {
        CudaOptions parsed;
        if (variant == "default") {
            return parsed;
        }
        const std::string_view head = variant.substr(0, variant.find('+'));
        std::string_view suffixes = variant.substr(head.size());
        bool table = false;
        bool graph = false;
        bool async = false;
        while (!suffixes.empty()) {
            suffixes.remove_prefix(1);
            const std::string_view suffix = suffixes.substr(0, suffixes.find('+'));
            suffixes.remove_prefix(suffix.size());
            if (suffix == "table" && !table) {
                table = true;
            } else if (suffix == "graph" && !graph) {
                graph = true;
            } else if (suffix == "async" && !async) {
                async = true;
            } else {
                malformed(variant, "expected +table, +graph or +async, each once");
            }
        }
        parsed.coeffs = table ? CudaCoeffs::Table : CudaCoeffs::Arrays;
        parsed.graph = graph;
        parsed.async_snapshots = async;

        const std::size_t colon = head.find(':');
        const std::string_view kernel = head.substr(0, colon);
        if (kernel == "fused") {
            parsed.kernel = CudaKernel::Fused;
        } else if (kernel == "tiled") {
            parsed.kernel = CudaKernel::Tiled;
        } else {
            malformed(variant, "expected fused or tiled");
        }
        if (colon != std::string_view::npos) {
            const std::string_view size = head.substr(colon + 1);
            const std::size_t cross = size.find('x');
            if (cross == std::string_view::npos) {
                malformed(variant, "expected a block size as WxH");
            }
            parsed.block_x = parse_side(variant, size.substr(0, cross));
            parsed.block_y = parse_side(variant, size.substr(cross + 1));
        }
        return parsed;
    }

    // Write the kernel and the block, then the suffixes.
    std::string CudaOptions::name() const
    {
        std::ostringstream text;
        text << kernel_name(kernel);
        text << ':';
        text << block_x;
        text << 'x';
        text << block_y;
        if (coeffs == CudaCoeffs::Table) {
            text << "+table";
        }
        if (graph) {
            text << "+graph";
        }
        if (async_snapshots) {
            text << "+async";
        }
        return text.str();
    }
}
