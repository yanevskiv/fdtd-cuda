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

// The device queries and the check that turns a failed call into a CudaError.
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_check.cuh"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_device.h"

// CUDA runtime.
#include <cuda_runtime.h>

// The table of distinct coefficients.
#include "fdtd/TMz/fdtd_tmz_coeff_table.h"

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
        // Threads per block of the kernels over a list of cells: sources, probes, slabs and edge.
        constexpr unsigned LINE_BLOCK = 128;

        // Most blocks a launch may have along y.
        constexpr std::size_t MAX_BLOCKS_Y = 65535;

        // Steps a CUDA graph holds.
        constexpr std::size_t GRAPH_STEPS = 16;

        // Column of the thread along x in the launch.
        __device__ inline int thread_i()
        {
            return static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        }

        // Row of the thread along y in the launch.
        __device__ inline int thread_j()
        {
            return static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        }

        // Advance Hx and Hy of the thread's cell.
        __global__ void update_h_kernel(gpu::DeviceFields fields, gpu::DeviceCoeffs coeffs)
        {
            const int i = thread_i();
            const int j = thread_j();
            if (i < fields.nx && j < fields.ny) {
                gpu::update_h_cell(fields, coeffs, i, j);
            }
        }

        // Advance Ez of the thread's cell by the edge, PML or interior update.
        __global__ void update_e_fused_kernel(gpu::DeviceFields fields, gpu::DeviceCoeffs coeffs,
                                              Boundary boundary)
        {
            const int i = thread_i();
            const int j = thread_j();
            if (i < fields.nx && j < fields.ny) {
                gpu::update_e_cell(fields, coeffs, i, j, boundary);
            }
        }

        // Advance Ez of the thread's interior cell, the interior starting at lo along both axes.
        __global__ void update_e_interior_kernel(gpu::DeviceFields fields, gpu::DeviceCoeffs coeffs,
                                                 int lo)
        {
            const int i = lo + thread_i();
            const int j = lo + thread_j();
            if (i < fields.nx - lo && j < fields.ny - lo) {
                gpu::update_e_interior_cell(fields, coeffs, i, j);
            }
        }

        // Advance Ezx, Ezy and Ez of the thread's PML cell.
        __global__ void update_e_slab_kernel(gpu::DeviceFields fields, gpu::DeviceCoeffs coeffs,
                                             int count)
        {
            const int t = thread_i();
            if (t < count) {
                const gpu::CellIndex cell = gpu::slab_cell(fields, t);
                gpu::update_e_split_cell(fields, coeffs, cell.i, cell.j);
            }
        }

        // Update Ez of the thread's edge cell.
        __global__ void update_e_edge_kernel(gpu::DeviceFields fields, gpu::DeviceCoeffs coeffs,
                                             int count, Boundary boundary)
        {
            const int t = thread_i();
            if (t < count) {
                const gpu::CellIndex cell = gpu::edge_cell(fields, t);
                gpu::update_e_edge_cell(fields, coeffs, cell.i, cell.j, boundary);
            }
        }

        // Row of a step in a step-major array, the step read from the device when it is there.
        __device__ inline std::ptrdiff_t step_row(const int *step, int stride)
        {
            return step != nullptr ? static_cast<std::ptrdiff_t>(*step) * stride : 0;
        }

        // Add one step's value to the thread's source cell. With a step on the device, the values
        // are those of all steps and the step picks the row, so a graph can replay the kernel.
        __global__ void add_sources_kernel(gpu::DeviceFields fields, const gpu::DevicePoint *points,
                                           int count, const FloatType *values, const int *step,
                                           int stride)
        {
            const int t = thread_i();
            if (t < count) {
                const gpu::DevicePoint point = points[t];
                const FloatType value = values[step_row(step, stride) + point.column];
                gpu::field_of(fields, point.component)[point.cell] += value;
            }
        }

        // Copy the thread's probe cell into one step's records, the step picking the row as for
        // the sources.
        __global__ void record_probes_kernel(gpu::DeviceFields fields,
                                             const gpu::DevicePoint *points, int count,
                                             FloatType *records, const int *step, int stride)
        {
            const int t = thread_i();
            if (t < count) {
                const gpu::DevicePoint point = points[t];
                records[step_row(step, stride) + point.column] =
                    gpu::field_of(fields, point.component)[point.cell];
            }
        }

        // Count a step on the device.
        __global__ void next_step_kernel(int *step)
        {
            ++*step;
        }

        // Set the step on the device.
        __global__ void set_step_kernel(int *step, int value)
        {
            *step = value;
        }

        // Add to a cell's new H the values of the magnetic sources on it, as add_sources_kernel
        // would once the H update is done.
        __device__ inline void add_h_sources(gpu::HValues &h, int k, const gpu::DevicePoint *points,
                                             int count, const FloatType *values, const int *step,
                                             int stride)
        {
            for (int s = 0; s < count; ++s) {
                const gpu::DevicePoint point = points[s];
                if (point.cell == k) {
                    const FloatType value = values[step_row(step, stride) + point.column];
                    if (point.component == Component::Hx) {
                        h.hx += value;
                    } else {
                        h.hy += value;
                    }
                }
            }
        }

        // Advance H and E of the thread's cell in one pass. The block updates H on its tile, and
        // on the column left of it and the row below it, which the neighbouring blocks own and
        // update the same way; the magnetic sources follow, then the tile's E from the new H in
        // shared memory. Ez, Hx and Hy are read from the fields and written to the next arrays,
        // so no block reads what another writes; Ezx and Ezy are updated in place.
        __global__ void update_tiled_kernel(gpu::DeviceFields fields, gpu::DeviceCoeffs coeffs,
                                            Boundary boundary, const gpu::DevicePoint *points,
                                            int magnetic, const FloatType *values, const int *step,
                                            int stride)
        {
            extern __shared__ FloatType tile[];
            const int width = static_cast<int>(blockDim.x) + 1;
            FloatType *tileHx = tile;
            FloatType *tileHy = tile + width * (static_cast<int>(blockDim.y) + 1);
            const int c =
                (static_cast<int>(threadIdx.y) + 1) * width + static_cast<int>(threadIdx.x) + 1;
            const int i = thread_i();
            const int j = thread_j();
            const bool inside = i < fields.nx && j < fields.ny;
            const int k = j * fields.pitch + i;
            if (inside) {
                gpu::HValues h = gpu::h_values(fields, coeffs, i, j);
                add_h_sources(h, k, points, magnetic, values, step, stride);
                tileHx[c] = h.hx;
                tileHy[c] = h.hy;
                fields.hx_next[k] = h.hx;
                fields.hy_next[k] = h.hy;
                if (threadIdx.x == 0 && i > 0) {
                    gpu::HValues left = gpu::h_values(fields, coeffs, i - 1, j);
                    add_h_sources(left, k - 1, points, magnetic, values, step, stride);
                    tileHy[c - 1] = left.hy;
                }
                if (threadIdx.y == 0 && j > 0) {
                    gpu::HValues below = gpu::h_values(fields, coeffs, i, j - 1);
                    add_h_sources(below, k - fields.pitch, points, magnetic, values, step, stride);
                    tileHx[c - width] = below.hx;
                }
            }
            __syncthreads();
            if (inside) {
                const gpu::HAround h{
                    i + 1 < fields.nx ? tileHy[c] : FloatType(0),
                    i > 0 ? tileHy[c - 1] : FloatType(0),
                    j + 1 < fields.ny ? tileHx[c] : FloatType(0),
                    j > 0 ? tileHx[c - width] : FloatType(0),
                };
                fields.ez_next[k] = gpu::e_value(fields, coeffs, i, j, boundary, h);
            }
        }

        // Blocks of LINE_BLOCK threads that cover a number of cells.
        unsigned line_blocks(std::size_t count)
        {
            return static_cast<unsigned>((count + LINE_BLOCK - 1) / LINE_BLOCK);
        }

        // Check the options and the grid before anything is allocated.
        const Simulation::Options &checked(const Grid &grid, const Simulation::Options &options)
        {
            if (gpu::device_count() == 0) {
                throw std::runtime_error("CudaBackend: no CUDA device");
            }
            const std::size_t bx = options.cuda_block_x;
            const std::size_t by = options.cuda_block_y;
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

        // Grow a device buffer to hold at least a number of elements, dropping its contents.
        template <typename T> void reserve(gpu::DeviceBuffer<T> &buffer, std::size_t size)
        {
            if (buffer.size() < size) {
                buffer = gpu::DeviceBuffer<T>(size);
            }
        }

        // Check the launches queued since the last check.
        void check_launch()
        {
            gpu::check(cudaGetLastError(), "kernel launch");
        }

        // Copy of a table entry, or none for a null one.
        std::optional<CoeffTuple> tuple_of(const CoeffTuple *entry)
        {
            if (entry == nullptr) {
                return std::nullopt;
            }
            return *entry;
        }
    }

    // Check the options, allocate the padded arrays and zero them.
    CudaBackend::CudaBackend(const Grid &grid, const Simulation::Options &options)
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
        if (m_options.cuda_kernel == CudaKernel::Tiled) {
            for (gpu::DeviceBuffer<FloatType> *buffer : {&m_ez_next, &m_hx_next, &m_hy_next}) {
                *buffer = gpu::DeviceBuffer<FloatType>(m_pitch * grid.ny());
                buffer->zero(m_stream);
            }
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
        if (m_options.cuda_coeffs == CudaCoeffs::Table) {
            const CoeffTable table(coeffs);
            reserve(m_table_entries, std::max<std::size_t>(table.entries().size(), 1));
            gpu::upload(m_table_entries, table.entries(), m_stream);
            reserve(m_table_index, m_pitch * ny);
            gpu::upload_rows(m_table_index, m_pitch, table.index(), host, nx, ny, m_stream);
            m_uniform_pml =
                tuple_of(table.uniform_entry(std::max<std::size_t>(coeffs.pml_thickness(), 1)));
            m_uniform_plain = tuple_of(table.uniform_entry(1));
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

    // Model the kernels: Fused and Split make an H pass and an E pass, Tiled one pass.
    StepTraffic CudaBackend::step_traffic() const noexcept
    {
        TrafficModel model;
        model.one_pass = m_options.cuda_kernel == CudaKernel::Tiled;
        if (m_options.cuda_coeffs == CudaCoeffs::Table) {
            model.table = true;
            model.index = !uniform_tuple().has_value();
        }
        return two_pass_traffic(m_grid, m_boundary == Boundary::Pml ? m_pml_thickness : 0,
                                m_boundary, model);
    }

    // Pick the interior's tuple for the boundary in force.
    const std::optional<CoeffTuple> &CudaBackend::uniform_tuple() const noexcept
    {
        return m_boundary == Boundary::Pml ? m_uniform_pml : m_uniform_plain;
    }

    // Return the options.
    const Simulation::Options &CudaBackend::options() const noexcept
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
        reserve(m_device_probes, device.size());
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

    // Copy each snapshot out on the copy stream while the steps after it run, and hand it over
    // once the next one is queued, so the handler runs while the device steps.
    void CudaBackend::advance_snapshots(std::size_t nsteps, std::span<const PointInjection> points,
                                        std::span<const FloatType> values,
                                        std::span<const std::size_t> snapshots,
                                        const SnapshotHandler &handler)
    {
        if (!m_options.cuda_async_snapshots || snapshots.empty()) {
            Backend::advance_snapshots(nsteps, points, values, snapshots, handler);
            return;
        }
        require_coeffs();
        const std::size_t magnetic = put_sources(nsteps, points, values);
        const std::size_t electric = points.size() - magnetic;
        const gpu::DeviceCoeffs coeffs = device_coeffs();
        if (!m_async) {
            m_async = std::make_unique<AsyncSnapshots>(m_grid, m_pitch);
        }
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

    // Copy the source cells and values to the device, magnetic cells first, and make room for
    // the records.
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
        reserve(m_device_sources, sources.size());
        gpu::upload(m_device_sources, std::span<const gpu::DevicePoint>(sources), m_stream);
        reserve(m_device_values, values.size());
        gpu::upload(m_device_values, values, m_stream);
        reserve(m_device_records, nsteps * m_probes.size());
        return static_cast<std::size_t>(firstElectric - sources.begin());
    }

    // Replay the graph over whole runs of GRAPH_STEPS steps when it is on, and launch the rest
    // step by step.
    void CudaBackend::queue_steps(const gpu::DeviceCoeffs &coeffs, std::size_t from, std::size_t to,
                                  std::size_t magnetic, std::size_t electric)
    {
        std::size_t n = from;
        if (m_options.cuda_graph && to - from >= GRAPH_STEPS) {
            const GraphKey key{
                .magnetic = magnetic,
                .electric = electric,
                .probes = m_probes.size(),
                .boundary = m_boundary,
                .pml = device_fields().pml,
                .ez = m_ez.data(),
                .sources = m_device_sources.data(),
                .values = m_device_values.data(),
                .points = m_device_probes.data(),
                .records = m_device_records.data(),
            };
            if (m_graph.empty() || !(key == m_graph_key)) {
                m_graph.capture(m_stream, [&] {
                    for (std::size_t s = 0; s < GRAPH_STEPS; ++s) {
                        launch_step(coeffs, 0, magnetic, electric, true);
                    }
                });
                m_graph_key = key;
            }
            set_step_kernel<<<1, 1, 0, m_stream.handle()>>>(m_device_step.data(),
                                                            static_cast<int>(from));
            check_launch();
            for (; n + GRAPH_STEPS <= to; n += GRAPH_STEPS) {
                m_graph.launch(m_stream);
            }
        }
        for (; n < to; ++n) {
            launch_step(coeffs, n, magnetic, electric, false);
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

    // Copy the fields to the slot's device arrays on the stream, then to its host fields on the
    // copy stream once that is done.
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

    // View the device coefficients.
    gpu::DeviceCoeffs CudaBackend::device_coeffs() const noexcept
    {
        gpu::DeviceCoeffs coeffs{
            .coeff_ez_x1 = m_coeff_ez_x1.data(),
            .coeff_ez_x2 = m_coeff_ez_x2.data(),
            .coeff_ez_y1 = m_coeff_ez_y1.data(),
            .coeff_ez_y2 = m_coeff_ez_y2.data(),
            .coeff_hx_1 = m_coeff_hx_1.data(),
            .coeff_hx_2 = m_coeff_hx_2.data(),
            .coeff_hy_1 = m_coeff_hy_1.data(),
            .coeff_hy_2 = m_coeff_hy_2.data(),
        };
        if (m_options.cuda_coeffs == CudaCoeffs::Table) {
            const std::optional<CoeffTuple> &uniform = uniform_tuple();
            if (uniform.has_value()) {
                coeffs.source = gpu::CoeffSource::Uniform;
                coeffs.uniform = *uniform;
            } else {
                coeffs.source = gpu::CoeffSource::Indexed;
                coeffs.table = m_table_entries.data();
                coeffs.index = m_table_index.data();
            }
        }
        return coeffs;
    }

    // Queue the interior, slab and edge kernels of the Split layout.
    void CudaBackend::launch_split_e(const gpu::DeviceFields &fields,
                                     const gpu::DeviceCoeffs &coeffs)
    {
        cudaStream_t stream = m_stream.handle();
        const int lo = gpu::interior_lo(fields);
        if (fields.nx > 2 * lo && fields.ny > 2 * lo) {
            const auto bx = static_cast<unsigned>(m_options.cuda_block_x);
            const auto by = static_cast<unsigned>(m_options.cuda_block_y);
            const auto width = static_cast<unsigned>(fields.nx - 2 * lo);
            const auto height = static_cast<unsigned>(fields.ny - 2 * lo);
            const dim3 block(bx, by);
            const dim3 cells((width + bx - 1) / bx, (height + by - 1) / by);
            update_e_interior_kernel<<<cells, block, 0, stream>>>(fields, coeffs, lo);
        }
        const int slabs = gpu::slab_cell_count(fields);
        if (slabs > 0) {
            update_e_slab_kernel<<<line_blocks(static_cast<std::size_t>(slabs)), LINE_BLOCK, 0,
                                   stream>>>(fields, coeffs, slabs);
        }
        const int edges = gpu::edge_cell_count(fields);
        update_e_edge_kernel<<<line_blocks(static_cast<std::size_t>(edges)), LINE_BLOCK, 0,
                               stream>>>(fields, coeffs, edges, m_boundary);
    }

    // Queue H, the H sources, E, the Ez sources and the probes of step n, or of the step on the
    // device, which is then counted. The tiled kernel does H, its sources and E in one pass and
    // swaps the arrays it read with those it wrote.
    void CudaBackend::launch_step(const gpu::DeviceCoeffs &coeffs, std::size_t n,
                                  std::size_t magnetic, std::size_t electric, bool replay)
    {
        const gpu::DeviceFields fields = device_fields();
        cudaStream_t stream = m_stream.handle();
        const auto bx = static_cast<unsigned>(m_options.cuda_block_x);
        const auto by = static_cast<unsigned>(m_options.cuda_block_y);
        const dim3 block(bx, by);
        const dim3 cells(static_cast<unsigned>((m_grid.nx() + bx - 1) / bx),
                         static_cast<unsigned>((m_grid.ny() + by - 1) / by));
        const std::size_t npoints = magnetic + electric;
        const std::size_t nprobes = m_probes.size();
        const FloatType *values = m_device_values.data() + (replay ? 0 : n * npoints);
        FloatType *records = m_device_records.data() + (replay ? 0 : n * nprobes);
        const int *step = replay ? m_device_step.data() : nullptr;
        const gpu::DevicePoint *sources = m_device_sources.data();

        gpu::DeviceFields after = fields;
        if (m_options.cuda_kernel == CudaKernel::Tiled) {
            const std::size_t shared = 2 * (bx + 1) * (by + 1) * sizeof(FloatType);
            update_tiled_kernel<<<cells, block, shared, stream>>>(
                fields, coeffs, m_boundary, sources, static_cast<int>(magnetic), values, step,
                static_cast<int>(npoints));
            std::swap(m_ez, m_ez_next);
            std::swap(m_hx, m_hx_next);
            std::swap(m_hy, m_hy_next);
            after = device_fields();
        } else {
            update_h_kernel<<<cells, block, 0, stream>>>(fields, coeffs);
            if (magnetic > 0) {
                add_sources_kernel<<<line_blocks(magnetic), LINE_BLOCK, 0, stream>>>(
                    fields, sources, static_cast<int>(magnetic), values, step,
                    static_cast<int>(npoints));
            }
            if (m_options.cuda_kernel == CudaKernel::Fused) {
                update_e_fused_kernel<<<cells, block, 0, stream>>>(fields, coeffs, m_boundary);
            } else {
                launch_split_e(fields, coeffs);
            }
        }
        if (electric > 0) {
            add_sources_kernel<<<line_blocks(electric), LINE_BLOCK, 0, stream>>>(
                after, sources + magnetic, static_cast<int>(electric), values, step,
                static_cast<int>(npoints));
        }
        if (nprobes > 0) {
            record_probes_kernel<<<line_blocks(nprobes), LINE_BLOCK, 0, stream>>>(
                after, m_device_probes.data(), static_cast<int>(nprobes), records, step,
                static_cast<int>(nprobes));
        }
        if (replay && npoints + nprobes > 0) {
            next_step_kernel<<<1, 1, 0, stream>>>(m_device_step.data());
        }
        check_launch();
    }
}
