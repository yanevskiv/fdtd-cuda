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

// Standard library for partitions, limits, sizes, string building, errors and buffers.
#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

// Namespace for the 2D TMz backends.
namespace fdtd::TMz::backends
{
    // Helpers private to this file.
    namespace
    {
        // Threads per block of the source and probe kernels.
        constexpr unsigned POINT_BLOCK = 128;

        // Most blocks a launch may have along y.
        constexpr std::size_t MAX_BLOCKS_Y = 65535;

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

        // Add one step's value to the thread's source cell.
        __global__ void add_sources_kernel(gpu::DeviceFields fields, const gpu::DevicePoint *points,
                                           int count, const FloatType *values)
        {
            const int t = thread_i();
            if (t < count) {
                const gpu::DevicePoint point = points[t];
                gpu::field_of(fields, point.component)[point.cell] += values[point.column];
            }
        }

        // Copy the thread's probe cell into one step's records.
        __global__ void record_probes_kernel(gpu::DeviceFields fields,
                                             const gpu::DevicePoint *points, int count,
                                             FloatType *records)
        {
            const int t = thread_i();
            if (t < count) {
                const gpu::DevicePoint point = points[t];
                records[point.column] = gpu::field_of(fields, point.component)[point.cell];
            }
        }

        // Blocks of POINT_BLOCK threads that cover a number of points.
        unsigned point_blocks(std::size_t count)
        {
            return static_cast<unsigned>((count + POINT_BLOCK - 1) / POINT_BLOCK);
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
    {
        for (gpu::DeviceBuffer<FloatType> *buffer :
             {&m_ez, &m_hx, &m_hy, &m_ezx, &m_ezy, &m_coeff_ez_x1, &m_coeff_ez_x2, &m_coeff_ez_y1,
              &m_coeff_ez_y2, &m_coeff_hx_1, &m_coeff_hx_2, &m_coeff_hy_1, &m_coeff_hy_2}) {
            buffer->zero(m_stream);
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
        m_stream.synchronize();
        m_pml_thickness = coeffs.pml_thickness();
        m_has_coeffs = true;
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
        std::vector<gpu::DevicePoint> sources;
        sources.reserve(points.size());
        for (std::size_t k = 0; k < points.size(); ++k) {
            sources.push_back({
                .component = points[k].component,
                .cell = device_index(points[k].index),
                .column = static_cast<int>(k),
            });
        }
        const auto electric =
            std::stable_partition(sources.begin(), sources.end(), [](const gpu::DevicePoint &p) {
                return p.component != Component::Ez;
            });
        const auto magnetic = static_cast<std::size_t>(electric - sources.begin());
        reserve(m_device_sources, sources.size());
        gpu::upload(m_device_sources, std::span<const gpu::DevicePoint>(sources), m_stream);
        reserve(m_device_values, values.size());
        gpu::upload(m_device_values, values, m_stream);
        reserve(m_device_records, nsteps * m_probes.size());

        const gpu::DeviceFields fields = device_fields();
        const gpu::DeviceCoeffs coeffs = device_coeffs();
        for (std::size_t n = 0; n < nsteps; ++n) {
            launch_step(fields, coeffs, n, magnetic, sources.size() - magnetic);
        }

        const std::size_t first = m_records.size();
        m_records.resize(first + nsteps * m_probes.size());
        gpu::download(std::span<FloatType>(m_records).subspan(first), m_device_records, m_stream);
        m_stream.synchronize();
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
        };
    }

    // View the device coefficients.
    gpu::DeviceCoeffs CudaBackend::device_coeffs() const noexcept
    {
        return {
            .coeff_ez_x1 = m_coeff_ez_x1.data(),
            .coeff_ez_x2 = m_coeff_ez_x2.data(),
            .coeff_ez_y1 = m_coeff_ez_y1.data(),
            .coeff_ez_y2 = m_coeff_ez_y2.data(),
            .coeff_hx_1 = m_coeff_hx_1.data(),
            .coeff_hx_2 = m_coeff_hx_2.data(),
            .coeff_hy_1 = m_coeff_hy_1.data(),
            .coeff_hy_2 = m_coeff_hy_2.data(),
        };
    }

    // Queue H, the H sources, E, the Ez sources and the probes of step n.
    void CudaBackend::launch_step(const gpu::DeviceFields &fields, const gpu::DeviceCoeffs &coeffs,
                                  std::size_t n, std::size_t magnetic, std::size_t electric)
    {
        cudaStream_t stream = m_stream.handle();
        const auto bx = static_cast<unsigned>(m_options.cuda_block_x);
        const auto by = static_cast<unsigned>(m_options.cuda_block_y);
        const dim3 block(bx, by);
        const dim3 cells(static_cast<unsigned>((m_grid.nx() + bx - 1) / bx),
                         static_cast<unsigned>((m_grid.ny() + by - 1) / by));
        const std::size_t npoints = magnetic + electric;
        const FloatType *values = m_device_values.data() + n * npoints;
        const gpu::DevicePoint *sources = m_device_sources.data();

        update_h_kernel<<<cells, block, 0, stream>>>(fields, coeffs);
        if (magnetic > 0) {
            add_sources_kernel<<<point_blocks(magnetic), POINT_BLOCK, 0, stream>>>(
                fields, sources, static_cast<int>(magnetic), values);
        }
        update_e_fused_kernel<<<cells, block, 0, stream>>>(fields, coeffs, m_boundary);
        if (electric > 0) {
            add_sources_kernel<<<point_blocks(electric), POINT_BLOCK, 0, stream>>>(
                fields, sources + magnetic, static_cast<int>(electric), values);
        }
        const std::size_t nprobes = m_probes.size();
        if (nprobes > 0) {
            record_probes_kernel<<<point_blocks(nprobes), POINT_BLOCK, 0, stream>>>(
                fields, m_device_probes.data(), static_cast<int>(nprobes),
                m_device_records.data() + n * nprobes);
        }
        check_launch();
    }
}
