/*
 * C++ file for the 2D TMz backend interface and registry.
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

// Interface declaration.
#include "fdtd/TMz/fdtd_tmz_backend.h"

// Backend implementations.
#include "fdtd/TMz/serial/fdtd_tmz_serial_backend.h"

// Standard library for algorithms, errors and strings.
#include <algorithm>
#include <stdexcept>
#include <string>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Release nothing; derived backends own their resources.
    Backend::~Backend() = default;

    // Map a backend type to its name.
    std::string_view backend_name(BackendType type) noexcept
    {
        switch (type) {
            case BackendType::Serial:
                return "serial";
            case BackendType::OpenMP:
                return "openmp";
            case BackendType::Cuda:
                return "cuda";
        }
        return "unknown";
    }

    // List the backends compiled into this build.
    std::vector<BackendType> available_backends()
    {
        return {BackendType::Serial};
    }

    // Check whether a backend is compiled into this build.
    bool is_backend_available(BackendType type)
    {
        const auto types = available_backends();
        return std::find(types.begin(), types.end(), type) != types.end();
    }

    // Create a backend, or report that it is not compiled in.
    std::unique_ptr<Backend> make_backend(BackendType type, const Grid2D &grid)
    {
        switch (type) {
            case BackendType::Serial:
                return std::make_unique<SerialBackend>(grid);
            case BackendType::OpenMP:
            case BackendType::Cuda:
                break;
        }
        throw std::runtime_error("make_backend: the " + std::string(backend_name(type)) +
                                 " backend is not compiled into this build");
    }
}
