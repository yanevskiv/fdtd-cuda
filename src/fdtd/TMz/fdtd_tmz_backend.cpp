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

// The CUDA device count, for the registry.
#if FDTD_WITH_CUDA
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_device.h"
#endif

// Standard library for algorithms, limits, errors and pairs.
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Check that a flat index names a cell inside the grid, not padding.
    static bool inside(const Grid &grid, std::size_t index)
    {
        return index % grid.pitch() < grid.nx() && index / grid.pitch() < grid.ny();
    }

    // Release nothing; derived backends own their resources.
    Backend::~Backend() = default;

    // Advance with no sources.
    void Backend::step(std::size_t nsteps)
    {
        advance(nsteps, {}, {});
    }

    // Check the values, cells and distinctness, then advance.
    void Backend::step(std::size_t nsteps, std::span<const PointInjection> points,
                       std::span<const FloatType> values)
    {
        const std::size_t npoints = points.size();
        if (npoints != 0 && nsteps > std::numeric_limits<std::size_t>::max() / npoints) {
            throw std::invalid_argument("Backend: too many source values");
        }
        if (values.size() != nsteps * npoints) {
            throw std::invalid_argument("Backend: need one source value per point per step");
        }

        const Grid &g = grid();
        std::vector<std::pair<int, std::size_t>> keys;
        keys.reserve(npoints);
        for (const PointInjection &point : points) {
            if (!inside(g, point.index)) {
                throw std::invalid_argument("Backend: source point outside the grid");
            }
            keys.emplace_back(static_cast<int>(point.component), point.index);
        }
        std::sort(keys.begin(), keys.end());
        if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
            throw std::invalid_argument("Backend: source points must be distinct");
        }

        advance(nsteps, points, values);
    }

    // Check the probe cells, then replace the probes.
    void Backend::set_probes(std::span<const PointProbe> points)
    {
        for (const PointProbe &point : points) {
            if (!inside(grid(), point.index)) {
                throw std::invalid_argument("Backend: probe point outside the grid");
            }
        }
        reset_probes(points);
    }

    // Count the cells of each kind and weigh them by the words and flops of their updates.
    StepTraffic two_pass_traffic(const Grid &grid, std::size_t pml, Boundary boundary,
                                 const TrafficModel &model)
    {
        const double nx = static_cast<double>(grid.nx());
        const double ny = static_cast<double>(grid.ny());
        const double hx = nx * (ny - 1);
        const double hy = (nx - 1) * ny;

        const double inner = std::max(nx - 2, 0.0) * std::max(ny - 2, 0.0);
        const double lo = static_cast<double>(std::max<std::size_t>(pml, 1));
        const double interior = std::max(nx - 2 * lo, 0.0) * std::max(ny - 2 * lo, 0.0);
        const double split = inner - interior;
        const double edge = nx * ny - inner;
        const bool pmc = boundary == Boundary::Pmc;

        const double core = std::max(nx - 2 * lo - 1, 0.0) * std::max(ny - 2 * lo - 1, 0.0);
        const double tabled = model.table ? core : 0;
        const double words = nx * ny + 4 * (hx + hy) - 4 * tabled + 6 * interior - 2 * tabled +
                             11 * split + (pmc ? 6 : 3) * edge;
        const double index = model.index ? 2 * nx * ny : 0;
        const double flops = 5 * (hx + hy) + 8 * interior + 11 * split + (pmc ? 8 : 0) * edge;
        return {.bytes = words * static_cast<double>(sizeof(FloatType)) + index, .flops = flops};
    }

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

    // List the backends compiled into this build, CUDA only when a device is present.
    std::vector<BackendType> available_backends()
    {
        std::vector<BackendType> types{BackendType::Serial};
        if (FDTD_WITH_OPENMP) {
            types.push_back(BackendType::OpenMP);
        }
#if FDTD_WITH_CUDA
        if (backends::gpu::device_count() > 0) {
            types.push_back(BackendType::Cuda);
        }
#endif
        return types;
    }

    // Check whether a backend is compiled into this build.
    bool is_backend_available(BackendType type)
    {
        const auto types = available_backends();
        return std::find(types.begin(), types.end(), type) != types.end();
    }
}
