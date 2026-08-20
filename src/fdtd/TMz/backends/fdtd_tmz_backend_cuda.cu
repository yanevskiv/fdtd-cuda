/*
 * C++ file for the 2D TMz CUDA backend and its kernels.
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

// Class declaration.
#include "fdtd/TMz/backends/fdtd_tmz_backend_cuda.h"

// The device queries, and the kernels over the source and probe cells.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_device.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_points.h"

// CUDA runtime.
#include <cuda_runtime.h>

// Standard library for partitions, limits, sizes, string building, errors and buffers.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Helpers private to this file.
    namespace
    {
        // Most blocks a launch may have along y.
        constexpr std::size_t MAX_BLOCKS_Y = 65535;

        // Bytes of device memory the grid's arrays take: fields, coefficients, and the options'.
        std::size_t device_bytes(std::size_t cells, const CudaOptions &options)
        {
            std::size_t arrays = 13;
            if (options.level == CudaLevel::V4) {
                arrays += 3;
            }
            if (options.async_snapshots) {
                arrays += 10;
            }
            const bool table = options.level >= CudaLevel::V3;
            const std::size_t index = table ? cells : 0;
            const std::size_t entries =
                table ? gpu::CoeffTable::CAPACITY * sizeof(gpu::CoeffTuple) : 0;
            return arrays * cells * sizeof(FloatType) + index + entries + sizeof(int);
        }

        // Write a byte count in MiB for a message.
        void put_mib(std::ostringstream &message, std::size_t bytes)
        {
            message << bytes / (std::size_t(1) << 20);
            message << " MiB";
        }

        // Check the options and the grid before anything is allocated.
        const CudaOptions &checked(const Grid &grid, const CudaOptions &options)
        {
            if (gpu::device_count() == 0) {
                throw std::runtime_error("CudaBackend: no CUDA device");
            }
            const std::size_t bx = options.block_x;
            const std::size_t by = options.block_y;
            if (bx == 0 || by == 0) {
                throw std::invalid_argument("CudaBackend: block sides must be positive");
            }
            const std::size_t most = gpu::max_threads_per_block();
            if (bx > most || by > most / bx) {
                std::ostringstream message;
                message << "CudaBackend: a block holds at most ";
                message << most;
                message << " threads";
                throw std::invalid_argument(message.str());
            }
            if ((grid.ny() + by - 1) / by > MAX_BLOCKS_Y) {
                throw std::invalid_argument("CudaBackend: too many blocks along y");
            }
            const std::size_t pitch = gpu::device_pitch(grid.nx(), sizeof(FloatType));
            const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
            if (pitch > limit / grid.ny()) {
                throw std::invalid_argument("CudaBackend: the grid is too large for int indices");
            }
            const std::size_t needed = device_bytes(pitch * grid.ny(), options);
            const std::size_t free = gpu::free_memory();
            if (needed > free) {
                std::ostringstream message;
                message << "CudaBackend: the grid needs ";
                put_mib(message, needed);
                message << " of device memory, and ";
                put_mib(message, free);
                message << " are free";
                throw std::invalid_argument(message.str());
            }
            return options;
        }

        // Check that data is on the backend's grid.
        void require_same_grid(const Grid &expected, const Grid &actual)
        {
            if (expected.nx() != actual.nx() || expected.ny() != actual.ny() ||
                expected.pitch() != actual.pitch() || expected.dx() != actual.dx() ||
                expected.dy() != actual.dy()) {
                throw std::invalid_argument("cuda backend: the data is on a different grid");
            }
        }
    }

    // Check the options, allocate the padded arrays and zero them, and make the snapshot slots.
    CudaBackend::CudaBackend(const Grid &grid, const CudaOptions &options)
        : m_options(checked(grid, options))
        , m_grid(grid)
        , m_pitch(gpu::device_pitch(grid.nx(), sizeof(FloatType)))
        , m_ez(m_pitch * grid.ny())
        , m_hx(m_pitch * grid.ny())
        , m_hy(m_pitch * grid.ny())
        , m_ezx(m_pitch * grid.ny())
        , m_ezy(m_pitch * grid.ny())
        , m_coeff_ez_x1(m_pitch * grid.ny())
        , m_coeff_ez_x2(m_pitch * grid.ny())
        , m_coeff_ez_y1(m_pitch * grid.ny())
        , m_coeff_ez_y2(m_pitch * grid.ny())
        , m_coeff_hx_1(m_pitch * grid.ny())
        , m_coeff_hx_2(m_pitch * grid.ny())
        , m_coeff_hy_1(m_pitch * grid.ny())
        , m_coeff_hy_2(m_pitch * grid.ny())
        , m_device_step(1)
    {
        for (gpu::DeviceBuffer<FloatType> *buffer :
             {&m_ez, &m_hx, &m_hy, &m_ezx, &m_ezy, &m_coeff_ez_x1, &m_coeff_ez_x2, &m_coeff_ez_y1,
              &m_coeff_ez_y2, &m_coeff_hx_1, &m_coeff_hx_2, &m_coeff_hy_1, &m_coeff_hy_2}) {
            buffer->zero(m_stream);
        }
        if (m_options.level == CudaLevel::V4) {
            for (gpu::DeviceBuffer<FloatType> *buffer : {&m_ez_next, &m_hx_next, &m_hy_next}) {
                *buffer = gpu::DeviceBuffer<FloatType>(m_pitch * grid.ny());
                buffer->zero(m_stream);
            }
        }
        if (m_options.async_snapshots) {
            m_async = std::make_unique<AsyncSnapshots>(m_grid, m_pitch);
        }
        m_stream.synchronize();
    }

    // Return the backend name.
    std::string_view CudaBackend::name() const noexcept
    {
        return backend_name(BackendType::Cuda);
    }

    // Return the backend type.
    BackendType CudaBackend::type() const noexcept
    {
        return BackendType::Cuda;
    }

    // Return the grid.
    const Grid &CudaBackend::grid() const noexcept
    {
        return m_grid;
    }

    // Copy the five fields to the padded device rows.
    void CudaBackend::upload(const Fields &fields)
    {
        require_same_grid(m_grid, fields.grid());
        const std::size_t nx = m_grid.nx();
        const std::size_t ny = m_grid.ny();
        const std::size_t host = m_grid.pitch();
        gpu::upload_rows(m_ez, m_pitch, fields.ez(), host, nx, ny, m_stream);
        gpu::upload_rows(m_hx, m_pitch, fields.hx(), host, nx, ny, m_stream);
        gpu::upload_rows(m_hy, m_pitch, fields.hy(), host, nx, ny, m_stream);
        gpu::upload_rows(m_ezx, m_pitch, fields.ezx(), host, nx, ny, m_stream);
        gpu::upload_rows(m_ezy, m_pitch, fields.ezy(), host, nx, ny, m_stream);
        m_stream.synchronize();
    }

    // Copy the five fields from the padded device rows.
    void CudaBackend::download(Fields &fields) const
    {
        require_same_grid(m_grid, fields.grid());
        const std::size_t nx = m_grid.nx();
        const std::size_t ny = m_grid.ny();
        const std::size_t host = m_grid.pitch();
        gpu::download_rows(fields.ez(), host, m_ez, m_pitch, nx, ny, m_stream);
        gpu::download_rows(fields.hx(), host, m_hx, m_pitch, nx, ny, m_stream);
        gpu::download_rows(fields.hy(), host, m_hy, m_pitch, nx, ny, m_stream);
        gpu::download_rows(fields.ezx(), host, m_ezx, m_pitch, nx, ny, m_stream);
        gpu::download_rows(fields.ezy(), host, m_ezy, m_pitch, nx, ny, m_stream);
    }

    // Copy the eight coefficient arrays to the padded device rows.
    void CudaBackend::upload(const Coeffs &coeffs)
    {
        require_same_grid(m_grid, coeffs.grid());
        const std::size_t nx = m_grid.nx();
        const std::size_t ny = m_grid.ny();
        const std::size_t host = m_grid.pitch();
        gpu::upload_rows(m_coeff_ez_x1, m_pitch, coeffs.coeff_ez_x1(), host, nx, ny, m_stream);
        gpu::upload_rows(m_coeff_ez_x2, m_pitch, coeffs.coeff_ez_x2(), host, nx, ny, m_stream);
        gpu::upload_rows(m_coeff_ez_y1, m_pitch, coeffs.coeff_ez_y1(), host, nx, ny, m_stream);
        gpu::upload_rows(m_coeff_ez_y2, m_pitch, coeffs.coeff_ez_y2(), host, nx, ny, m_stream);
        gpu::upload_rows(m_coeff_hx_1, m_pitch, coeffs.coeff_hx_1(), host, nx, ny, m_stream);
        gpu::upload_rows(m_coeff_hx_2, m_pitch, coeffs.coeff_hx_2(), host, nx, ny, m_stream);
        gpu::upload_rows(m_coeff_hy_1, m_pitch, coeffs.coeff_hy_1(), host, nx, ny, m_stream);
        gpu::upload_rows(m_coeff_hy_2, m_pitch, coeffs.coeff_hy_2(), host, nx, ny, m_stream);
        if (m_options.level >= CudaLevel::V3) {
            upload_table(coeffs);
        }
        m_stream.synchronize();
        m_pml_thickness = coeffs.pml_thickness();
        m_has_coeffs = true;
        m_graph = gpu::Graph();
    }

    // Store the boundary condition.
    void CudaBackend::set_boundary(Boundary boundary)
    {
        m_boundary = boundary;
    }

    // Return the boundary condition.
    Boundary CudaBackend::boundary() const noexcept
    {
        return m_boundary;
    }

    // Return the number of probes.
    std::size_t CudaBackend::probe_count() const noexcept
    {
        return m_probes.size();
    }

    // Return the number of recorded steps.
    std::size_t CudaBackend::probe_steps() const noexcept
    {
        return m_probes.empty() ? 0 : m_records.size() / m_probes.size();
    }

    // Copy the records out.
    std::vector<FloatType> CudaBackend::fetch_probes() const
    {
        return m_records;
    }

    // Model the kernels: two passes below v4, one at v4, and the table from v3.
    StepTraffic CudaBackend::step_traffic() const noexcept
    {
        TrafficModel model;
        model.one_pass = m_options.level == CudaLevel::V4;
        if (m_options.level >= CudaLevel::V3) {
            model.table = true;
            model.index = !uniform_tuple().has_value();
        }
        return two_pass_traffic(m_grid, m_boundary == Boundary::Pml ? m_pml_thickness : 0,
                                m_boundary, model);
    }

    // Pick the interior's tuple for the boundary in force.
    const std::optional<gpu::CoeffTuple> &CudaBackend::uniform_tuple() const noexcept
    {
        return m_boundary == Boundary::Pml ? m_uniform_pml : m_uniform_plain;
    }

    // Return the options.
    const CudaOptions &CudaBackend::options() const noexcept
    {
        return m_options;
    }

    // Return the device pitch.
    std::size_t CudaBackend::device_pitch() const noexcept
    {
        return m_pitch;
    }

    // Keep the probe cells, put them on the device, and clear the records.
    void CudaBackend::reset_probes(std::span<const PointProbe> points)
    {
        std::vector<gpu::DevicePoint> device;
        device.reserve(points.size());
        for (std::size_t k = 0; k < points.size(); ++k) {
            device.push_back({
                .component = points[k].component,
                .cell = device_index(points[k].index),
                .column = static_cast<int>(k),
            });
        }
        gpu::reserve(m_device_probes, device.size());
        gpu::upload(m_device_probes, std::span<const gpu::DevicePoint>(device), m_stream);
        m_stream.synchronize();
        m_probes.assign(points.begin(), points.end());
        m_records.clear();
    }

    // Put the sources on the device, run the steps, and bring the records back.
    void CudaBackend::advance(std::size_t nsteps, std::span<const PointInjection> points,
                              std::span<const FloatType> values)
    {
        require_coeffs();
        const std::size_t magnetic = put_sources(nsteps, points, values);
        const std::size_t electric = points.size() - magnetic;
        queue_steps(device_coeffs(), 0, nsteps, magnetic, electric);
        collect_records(nsteps);
    }

    // Copy each snapshot out while the steps after it run, and hand it over after the next.
    void CudaBackend::advance_snapshots(std::size_t nsteps, std::span<const PointInjection> points,
                                        std::span<const FloatType> values,
                                        std::span<const std::size_t> snapshots,
                                        const SnapshotHandler &handler)
    {
        if (!m_options.async_snapshots || snapshots.empty()) {
            Backend::advance_snapshots(nsteps, points, values, snapshots, handler);
            return;
        }
        require_coeffs();
        const std::size_t magnetic = put_sources(nsteps, points, values);
        const std::size_t electric = points.size() - magnetic;
        const gpu::DeviceCoeffs coeffs = device_coeffs();
        std::size_t done = 0;
        for (std::size_t s = 0; s < snapshots.size(); ++s) {
            queue_steps(coeffs, done, snapshots[s], magnetic, electric);
            done = snapshots[s];
            queue_snapshot(m_async->slots[s % 2], done);
            if (s > 0) {
                deliver(m_async->slots[(s - 1) % 2], handler);
            }
        }
        queue_steps(coeffs, done, nsteps, magnetic, electric);
        deliver(m_async->slots[(snapshots.size() - 1) % 2], handler);
        collect_records(nsteps);
    }

    // Copy the sources to the device, magnetic first, and make room for the records.
    std::size_t CudaBackend::put_sources(std::size_t nsteps, std::span<const PointInjection> points,
                                         std::span<const FloatType> values)
    {
        std::vector<gpu::DevicePoint> sources;
        sources.reserve(points.size());
        for (std::size_t k = 0; k < points.size(); ++k) {
            sources.push_back({
                .component = points[k].component,
                .cell = device_index(points[k].index),
                .column = static_cast<int>(k),
            });
        }
        const auto firstElectric =
            std::stable_partition(sources.begin(), sources.end(), [](const gpu::DevicePoint &p) {
                return p.component != Component::Ez;
            });
        gpu::reserve(m_device_sources, sources.size());
        gpu::upload(m_device_sources, std::span<const gpu::DevicePoint>(sources), m_stream);
        gpu::reserve(m_device_values, values.size());
        gpu::upload(m_device_values, values, m_stream);
        gpu::reserve(m_device_records, nsteps * m_probes.size());
        return static_cast<std::size_t>(firstElectric - sources.begin());
    }

    // From v2 replay the graph over whole runs of its steps, and launch the rest one by one.
    void CudaBackend::queue_steps(const gpu::DeviceCoeffs &coeffs, std::size_t from, std::size_t to,
                                  std::size_t magnetic, std::size_t electric)
    {
        std::size_t n = from;
        if (m_options.level >= CudaLevel::V2) {
            n = queue_graph(coeffs, from, to, magnetic, electric);
        }
        for (; n < to; ++n) {
            launch_step(coeffs, n, magnetic, electric, Launch::Plain);
        }
    }

    // Append the records of the advance and wait for the device.
    void CudaBackend::collect_records(std::size_t nsteps)
    {
        const std::size_t first = m_records.size();
        m_records.resize(first + nsteps * m_probes.size());
        gpu::download(std::span<FloatType>(m_records).subspan(first), m_device_records, m_stream);
        m_stream.synchronize();
    }

    // Copy the fields to the slot's device arrays, then to its host fields on the copy stream.
    void CudaBackend::queue_snapshot(SnapshotSlot &slot, std::size_t steps)
    {
        const std::size_t bytes = m_pitch * m_grid.ny() * sizeof(FloatType);
        gpu::copy_on_device(slot.ez.data(), m_ez.data(), bytes, m_stream);
        gpu::copy_on_device(slot.hx.data(), m_hx.data(), bytes, m_stream);
        gpu::copy_on_device(slot.hy.data(), m_hy.data(), bytes, m_stream);
        gpu::copy_on_device(slot.ezx.data(), m_ezx.data(), bytes, m_stream);
        gpu::copy_on_device(slot.ezy.data(), m_ezy.data(), bytes, m_stream);
        slot.copied.record(m_stream);
        slot.copied.hold(m_async->stream);

        const std::size_t width = m_grid.nx() * sizeof(FloatType);
        const std::size_t host = m_grid.pitch() * sizeof(FloatType);
        const std::size_t device = m_pitch * sizeof(FloatType);
        const std::size_t ny = m_grid.ny();
        const gpu::Stream &copies = m_async->stream;
        gpu::queue_rows_to_host(slot.fields.ez().data(), host, slot.ez.data(), device, width, ny,
                                copies);
        gpu::queue_rows_to_host(slot.fields.hx().data(), host, slot.hx.data(), device, width, ny,
                                copies);
        gpu::queue_rows_to_host(slot.fields.hy().data(), host, slot.hy.data(), device, width, ny,
                                copies);
        gpu::queue_rows_to_host(slot.fields.ezx().data(), host, slot.ezx.data(), device, width, ny,
                                copies);
        gpu::queue_rows_to_host(slot.fields.ezy().data(), host, slot.ezy.data(), device, width, ny,
                                copies);
        slot.landed.record(copies);
        slot.steps = steps;
    }

    // Wait for the slot's copy to land and hand it over.
    void CudaBackend::deliver(const SnapshotSlot &slot, const SnapshotHandler &handler)
    {
        slot.landed.synchronize();
        handler(slot.steps, slot.fields);
    }

    // Allocate the device arrays and host fields of a slot, and pin the host fields.
    CudaBackend::SnapshotSlot::SnapshotSlot(const Grid &grid, std::size_t pitch)
        : ez(pitch * grid.ny())
        , hx(pitch * grid.ny())
        , hy(pitch * grid.ny())
        , ezx(pitch * grid.ny())
        , ezy(pitch * grid.ny())
        , fields(grid)
    {
        const std::size_t bytes = grid.size() * sizeof(FloatType);
        pins.emplace_back(fields.ez().data(), bytes);
        pins.emplace_back(fields.hx().data(), bytes);
        pins.emplace_back(fields.hy().data(), bytes);
        pins.emplace_back(fields.ezx().data(), bytes);
        pins.emplace_back(fields.ezy().data(), bytes);
    }

    // Make the copy stream and the two slots.
    CudaBackend::AsyncSnapshots::AsyncSnapshots(const Grid &grid, std::size_t pitch)
        : slots{SnapshotSlot(grid, pitch), SnapshotSlot(grid, pitch)}
    {
        // Empty.
    }

    // Throw unless coefficients were uploaded.
    void CudaBackend::require_coeffs() const
    {
        if (!m_has_coeffs) {
            throw std::logic_error("cuda backend: step before the coefficients were uploaded");
        }
    }

    // Move the cell from the host pitch to the device pitch.
    int CudaBackend::device_index(std::size_t index) const noexcept
    {
        const std::size_t i = index % m_grid.pitch();
        const std::size_t j = index / m_grid.pitch();
        return static_cast<int>(j * m_pitch + i);
    }

    // View the device fields, with the PML in force under Boundary::Pml only.
    gpu::DeviceFields CudaBackend::device_fields() noexcept
    {
        const std::size_t pml = m_boundary == Boundary::Pml ? m_pml_thickness : 0;
        return {
            .ez = m_ez.data(),
            .hx = m_hx.data(),
            .hy = m_hy.data(),
            .ezx = m_ezx.data(),
            .ezy = m_ezy.data(),
            .nx = static_cast<int>(m_grid.nx()),
            .ny = static_cast<int>(m_grid.ny()),
            .pitch = static_cast<int>(m_pitch),
            .inv_dx = FloatType(1) / m_grid.dx(),
            .inv_dy = FloatType(1) / m_grid.dy(),
            .pml = static_cast<int>(pml),
            .ez_next = m_ez_next.data(),
            .hx_next = m_hx_next.data(),
            .hy_next = m_hy_next.data(),
        };
    }

    // View the eight device arrays, with the table from v3.
    gpu::DeviceCoeffs CudaBackend::device_coeffs() const noexcept
    {
        const gpu::DeviceCoeffs arrays{
            .coeff_ez_x1 = m_coeff_ez_x1.data(),
            .coeff_ez_x2 = m_coeff_ez_x2.data(),
            .coeff_ez_y1 = m_coeff_ez_y1.data(),
            .coeff_ez_y2 = m_coeff_ez_y2.data(),
            .coeff_hx_1 = m_coeff_hx_1.data(),
            .coeff_hx_2 = m_coeff_hx_2.data(),
            .coeff_hy_1 = m_coeff_hy_1.data(),
            .coeff_hy_2 = m_coeff_hy_2.data(),
        };
        return m_options.level >= CudaLevel::V3 ? table_coeffs(arrays) : arrays;
    }

    // Queue the level's update of step n, or of the step on the device, then the Ez sources and
    // the probes, and count the step on the device for a replay.
    void CudaBackend::launch_step(const gpu::DeviceCoeffs &coeffs, std::size_t n,
                                  std::size_t magnetic, std::size_t electric, Launch launch)
    {
        const bool replay = launch == Launch::Replay;
        const std::size_t npoints = magnetic + electric;
        const std::size_t nprobes = m_probes.size();
        const StepPoints points{
            .sources = m_device_sources.data(),
            .magnetic = static_cast<int>(magnetic),
            .electric = static_cast<int>(electric),
            .values = m_device_values.data() + (replay ? 0 : n * npoints),
            .records = m_device_records.data() + (replay ? 0 : n * nprobes),
            .step = replay ? m_device_step.data() : nullptr,
        };
        const gpu::DeviceFields after = m_options.level == CudaLevel::V4
                                            ? launch_tiled(coeffs, points)
                                            : launch_two_kernels(coeffs, points);
        gpu::queue_add_sources(m_stream, after, points.sources + magnetic, points.electric,
                               points.values, points.step, static_cast<int>(npoints));
        gpu::queue_record_probes(m_stream, after, m_device_probes.data(), static_cast<int>(nprobes),
                                 points.records, points.step, static_cast<int>(nprobes));
        if (replay && npoints + nprobes > 0) {
            gpu::queue_count_step(m_stream, m_device_step.data());
        }
    }
}
