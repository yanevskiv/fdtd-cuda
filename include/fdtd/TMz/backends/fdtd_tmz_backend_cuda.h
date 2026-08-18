/*
 * C++ header for the 2D TMz CUDA backend.
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

#ifndef FDTD_TMZ_BACKEND_CUDA_H
#define FDTD_TMZ_BACKEND_CUDA_H

#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_buffer.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_cell.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_graph.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /**
             * GPU backend: the fields live in device memory and a step is a few kernels, one
             * thread per cell.
             *
             * Rows on the device are padded to 128 bytes; the host grid keeps its pitch, and
             * the copies and the cells of sources and probes go from one to the other. A step
             * is H, the Hx and Hy sources, E, the Ez sources and the probes, in that order on
             * one stream. E is one kernel or three, as cuda_kernel says. Each advance ends with
             * a wait for the stream, so the work is done when step() returns.
             */
            class CudaBackend final : public Backend
            {
            public:
                /**
                 * Create the backend with zeroed fields.
                 *
                 * @param grid Grid the backend works on.
                 * @param options Settings of a simulation; the backend reads the cuda_ fields.
                 * @throws std::runtime_error When no CUDA device is present.
                 * @throws std::invalid_argument When a block side is zero, the block holds more
                 *         threads than the device allows, the blocks along y pass the device's
                 *         limit of 65535, or the padded grid reaches 2^31 elements.
                 * @throws CudaError When the device memory cannot be had.
                 */
                CudaBackend(const Grid &grid, const Simulation::Options &options);

                /**
                 * Short lowercase name of the backend.
                 *
                 * @return "cuda".
                 */
                std::string_view name() const noexcept override;

                /**
                 * Type of the backend.
                 *
                 * @return BackendType::Cuda.
                 */
                BackendType type() const noexcept override;

                /**
                 * Grid the backend was made for.
                 *
                 * @return The grid, with the host pitch.
                 */
                const Grid &grid() const noexcept override;

                /**
                 * Copy fields to the device.
                 *
                 * @param fields Fields on the backend's grid.
                 */
                void upload(const Fields &fields) override;

                /**
                 * Copy the device's fields out.
                 *
                 * @param fields Destination on the backend's grid.
                 */
                void download(Fields &fields) const override;

                /**
                 * Copy update coefficients to the device.
                 *
                 * @param coeffs Coefficients on the backend's grid.
                 */
                void upload(const Coeffs &coeffs) override;

                /**
                 * Set the outer boundary condition.
                 *
                 * @param boundary Outer boundary condition.
                 */
                void set_boundary(Boundary boundary) override;

                /**
                 * Outer boundary condition.
                 *
                 * @return The boundary.
                 */
                Boundary boundary() const noexcept override;

                /**
                 * Number of probes set.
                 *
                 * @return The probe count.
                 */
                std::size_t probe_count() const noexcept override;

                /**
                 * Number of steps recorded since the probes were set.
                 *
                 * @return The recorded step count.
                 */
                std::size_t probe_steps() const noexcept override;

                /**
                 * Copy the probe records out, step-major.
                 *
                 * @return The records.
                 */
                std::vector<FloatType> fetch_probes() const override;

                /**
                 * Modelled traffic of one step, two passes over per-cell coefficient arrays.
                 *
                 * @return two_pass_traffic() of the grid, the active PML and the boundary.
                 */
                StepTraffic step_traffic() const noexcept override;

                /**
                 * Settings of the backend.
                 *
                 * @return The options it was made with.
                 */
                const Simulation::Options &options() const noexcept;

                /**
                 * Row pitch of the device arrays.
                 *
                 * @return Elements from one device row to the next.
                 */
                std::size_t device_pitch() const noexcept;

            protected:
                /**
                 * Replace the probes and clear the records.
                 *
                 * @param points Cells inside the grid.
                 */
                void reset_probes(std::span<const PointProbe> points) override;

                /**
                 * Advance the fields on the device and add the sources.
                 *
                 * @param nsteps Number of time steps.
                 * @param points Distinct cells inside the grid.
                 * @param values One value per point per step, step-major.
                 * @throws std::logic_error When no coefficients were uploaded.
                 * @throws CudaError When a kernel or a copy fails.
                 */
                void advance(std::size_t nsteps, std::span<const PointInjection> points,
                             std::span<const FloatType> values) override;

            private:
                /**
                 * Check that coefficients were uploaded, before a step.
                 *
                 * @throws std::logic_error When none were.
                 */
                void require_coeffs() const;

                /**
                 * Device index of a cell given by its host index.
                 *
                 * @param index Host index, j * nx + i.
                 * @return j * device_pitch() + i.
                 */
                int device_index(std::size_t index) const noexcept;

                /**
                 * View of the device fields for the kernels.
                 *
                 * @return The view, with the PML thickness of the boundary in force.
                 */
                gpu::DeviceFields device_fields() noexcept;

                /**
                 * View of the device coefficients for the kernels.
                 *
                 * @return The view.
                 */
                gpu::DeviceCoeffs device_coeffs() const noexcept;

                /**
                 * Queue the E update of the Split layout: the interior, the PML slabs, then the
                 * edge.
                 *
                 * @param fields View of the device fields.
                 * @param coeffs View of the device coefficients.
                 */
                void launch_split_e(const gpu::DeviceFields &fields,
                                    const gpu::DeviceCoeffs &coeffs);

                /**
                 * Run one step's kernels on the stream.
                 *
                 * @param fields View of the device fields.
                 * @param coeffs View of the device coefficients.
                 * @param n Step of this advance, from zero; unused when replay is set.
                 * @param magnetic Sources that add to Hx or Hy, first in the device sources.
                 * @param electric Sources that add to Ez, after the magnetic ones.
                 * @param replay Whether the kernels read the step from the device counter and
                 *        count it, so a graph can replay them, instead of being given n.
                 */
                void launch_step(const gpu::DeviceFields &fields, const gpu::DeviceCoeffs &coeffs,
                                 std::size_t n, std::size_t magnetic, std::size_t electric,
                                 bool replay);

                /** What a captured graph depends on besides the backend's fixed arrays. */
                struct GraphKey
                {
                    /** Sources that add to Hx or Hy. */
                    std::size_t magnetic = 0;

                    /** Sources that add to Ez. */
                    std::size_t electric = 0;

                    /** Probes. */
                    std::size_t probes = 0;

                    /** Outer boundary condition. */
                    Boundary boundary = Boundary::Pec;

                    /** PML thickness in force. */
                    int pml = 0;

                    /** Device sources. */
                    const void *sources = nullptr;

                    /** Device source values. */
                    const void *values = nullptr;

                    /** Device probe cells. */
                    const void *points = nullptr;

                    /** Device probe records. */
                    const void *records = nullptr;

                    /**
                     * Compare every member.
                     *
                     * @return Whether the keys are equal.
                     */
                    bool operator==(const GraphKey &) const = default;
                };

                /** Settings of the backend. */
                Simulation::Options m_options;

                /** Grid of the backend, with the host pitch. */
                Grid m_grid;

                /** Row pitch of the device arrays, in elements. */
                std::size_t m_pitch;

                /** Stream every copy and kernel is queued on. */
                gpu::Stream m_stream;

                /** Device Ez. */
                gpu::DeviceBuffer<FloatType> m_ez;

                /** Device Hx. */
                gpu::DeviceBuffer<FloatType> m_hx;

                /** Device Hy. */
                gpu::DeviceBuffer<FloatType> m_hy;

                /** Device Ezx. */
                gpu::DeviceBuffer<FloatType> m_ezx;

                /** Device Ezy. */
                gpu::DeviceBuffer<FloatType> m_ezy;

                /** Device Ez decay factors along x. */
                gpu::DeviceBuffer<FloatType> m_coeff_ez_x1;

                /** Device Ez curl factors along x. */
                gpu::DeviceBuffer<FloatType> m_coeff_ez_x2;

                /** Device Ez decay factors along y. */
                gpu::DeviceBuffer<FloatType> m_coeff_ez_y1;

                /** Device Ez curl factors along y. */
                gpu::DeviceBuffer<FloatType> m_coeff_ez_y2;

                /** Device Hx decay factors. */
                gpu::DeviceBuffer<FloatType> m_coeff_hx_1;

                /** Device Hx curl factors. */
                gpu::DeviceBuffer<FloatType> m_coeff_hx_2;

                /** Device Hy decay factors. */
                gpu::DeviceBuffer<FloatType> m_coeff_hy_1;

                /** Device Hy curl factors. */
                gpu::DeviceBuffer<FloatType> m_coeff_hy_2;

                /** Step of the advance that a replayed graph is on, counted on the device. */
                gpu::DeviceBuffer<int> m_device_step;

                /** Whether coefficients were uploaded. */
                bool m_has_coeffs = false;

                /** PML thickness the uploaded coefficients carry, used under Boundary::Pml. */
                std::size_t m_pml_thickness = 0;

                /** Outer boundary condition. */
                Boundary m_boundary = Boundary::Pec;

                /** Probe cells, as given. */
                std::vector<PointProbe> m_probes;

                /** Probe records, step-major. */
                std::vector<FloatType> m_records;

                /** Probe cells on the device, in the order given. */
                gpu::DeviceBuffer<gpu::DevicePoint> m_device_probes;

                /** Source cells of the current advance on the device, magnetic ones first. */
                gpu::DeviceBuffer<gpu::DevicePoint> m_device_sources;

                /** Source values of the current advance on the device, step-major. */
                gpu::DeviceBuffer<FloatType> m_device_values;

                /** Probe records of the current advance on the device, step-major. */
                gpu::DeviceBuffer<FloatType> m_device_records;

                /** Graph of GRAPH_STEPS steps, captured by the first advance that needs it. */
                gpu::Graph m_graph;

                /** What m_graph was captured with, captured again when this changes. */
                GraphKey m_graph_key;
            };
        }
    }
}

#endif
