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

#include "fdtd/TMz/backends/fdtd_tmz_backend_cuda_options.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_buffer.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_cell.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_event.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_graph.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_pinned.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"
#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_table.h"
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
             * one stream. Each advance ends with a wait for the stream, so the work is done when
             * step() returns.
             *
             * The level of the options adds one change at a time, each in its own file in
             * `backends/gpu/`: v1 the two kernels (`fdtd_tmz_backend_gpu_v1.cu`), v2 the graph
             * (`_v2.cu`), v3 the coefficient table (`_v3.cu`), v4 the tiled kernel (`_v4.cu`).
             * This file and `fdtd_tmz_backend_cuda.cu` hold what every level shares.
             */
            class CudaBackend final : public Backend
            {
            public:
                /**
                 * Create the backend with zeroed fields.
                 *
                 * @param grid Grid the backend works on.
                 * @param options Settings of the backend.
                 * @throws std::runtime_error When no CUDA device is present.
                 * @throws std::invalid_argument When a block side is zero, the block holds more
                 *         threads than the device allows, the blocks along y pass the device's
                 *         limit of 65535, the padded grid reaches 2^31 elements, or its arrays
                 *         need more device memory than is free; all before anything is
                 *         allocated.
                 * @throws gpu::CudaError When the device memory cannot be had.
                 */
                CudaBackend(const Grid &grid, const CudaOptions &options);

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
                 * Modelled traffic of one step of the kernel and coefficients in use.
                 *
                 * @return two_pass_traffic() of the grid, the active PML and the boundary, one
                 *         pass at v4, with the table from v3 and its index when the interior's
                 *         core does not share one tuple.
                 */
                StepTraffic step_traffic() const noexcept override;

                /**
                 * Settings of the backend.
                 *
                 * @return The options it was made with.
                 */
                const CudaOptions &options() const noexcept;

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
                 * @throws gpu::CudaError When a kernel or a copy fails.
                 */
                void advance(std::size_t nsteps, std::span<const PointInjection> points,
                             std::span<const FloatType> values) override;

                /**
                 * Advance the fields on the device, add the sources and hand over the snapshots.
                 *
                 * With async_snapshots, a snapshot is copied to device arrays of its own on
                 * the stream, then to page-locked host fields on a second stream, and handed
                 * over once the steps up to the next snapshot are queued, so the copy to the host
                 * and the handler run while the device steps. Two slots take turns. Without it,
                 * as Backend::advance_snapshots().
                 *
                 * @param nsteps Number of time steps.
                 * @param points Distinct cells inside the grid.
                 * @param values One value per point per step, step-major.
                 * @param snapshots Increasing step counts from 1 to nsteps.
                 * @param handler Receives each snapshot.
                 * @throws std::logic_error When no coefficients were uploaded.
                 * @throws gpu::CudaError When a kernel or a copy fails.
                 */
                void advance_snapshots(std::size_t nsteps, std::span<const PointInjection> points,
                                       std::span<const FloatType> values,
                                       std::span<const std::size_t> snapshots,
                                       const SnapshotHandler &handler) override;

            private:
                /** Steps a CUDA graph holds, from v2. */
                static constexpr std::size_t GRAPH_STEPS = 16;

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
                 * @return The view, with the table from v3 and the boundary in force.
                 */
                gpu::DeviceCoeffs device_coeffs() const noexcept;

                /**
                 * Tuple every interior cell shares under the boundary in force, from v3.
                 *
                 * @return The tuple, or none when the interior's cells differ.
                 */
                const std::optional<gpu::CoeffTuple> &uniform_tuple() const noexcept;

                /** A snapshot on its way out: its device copy, its host copy, and the events
                 * that mark the two copies done. */
                struct SnapshotSlot
                {
                    /**
                     * Allocate the arrays and fields, and page-lock the fields.
                     *
                     * @param grid Grid of the backend.
                     * @param pitch Row pitch of the device arrays.
                     */
                    SnapshotSlot(const Grid &grid, std::size_t pitch);

                    /** Device copy of Ez. */
                    gpu::DeviceBuffer<FloatType> ez;

                    /** Device copy of Hx. */
                    gpu::DeviceBuffer<FloatType> hx;

                    /** Device copy of Hy. */
                    gpu::DeviceBuffer<FloatType> hy;

                    /** Device copy of Ezx. */
                    gpu::DeviceBuffer<FloatType> ezx;

                    /** Device copy of Ezy. */
                    gpu::DeviceBuffer<FloatType> ezy;

                    /** Host copy, handed over. */
                    Fields fields;

                    /** Page locks of the host copy's arrays. */
                    std::vector<gpu::HostPin> pins;

                    /** Marks the device copy done. */
                    gpu::Event copied;

                    /** Marks the host copy done. */
                    gpu::Event landed;

                    /** Steps of the advance the snapshot was taken after. */
                    std::size_t steps = 0;
                };

                /** What async_snapshots needs, made with the backend so that no run pays
                 * for it. */
                struct AsyncSnapshots
                {
                    /**
                     * Make the copy stream and the slots.
                     *
                     * @param grid Grid of the backend.
                     * @param pitch Row pitch of the device arrays.
                     */
                    AsyncSnapshots(const Grid &grid, std::size_t pitch);

                    /** Stream the copies to the host run on. */
                    gpu::Stream stream;

                    /** Slots that take turns. */
                    SnapshotSlot slots[2];
                };

                /**
                 * Copy the source cells and values of an advance to the device, the magnetic
                 * cells first, and make room for its probe records.
                 *
                 * @param nsteps Number of time steps.
                 * @param points Distinct cells inside the grid.
                 * @param values One value per point per step, step-major.
                 * @return The number of magnetic sources.
                 */
                std::size_t put_sources(std::size_t nsteps, std::span<const PointInjection> points,
                                        std::span<const FloatType> values);

                /**
                 * Queue the steps from one step of the advance to another.
                 *
                 * From v2, whole runs of the graph's steps replay it (queue_graph()); the steps
                 * left over are launched one by one.
                 *
                 * @param coeffs View of the device coefficients.
                 * @param from First step, from zero.
                 * @param to Step to stop before.
                 * @param magnetic Sources that add to Hx or Hy.
                 * @param electric Sources that add to Ez.
                 */
                void queue_steps(const gpu::DeviceCoeffs &coeffs, std::size_t from, std::size_t to,
                                 std::size_t magnetic, std::size_t electric);

                /**
                 * Append the probe records of an advance and wait for the device.
                 *
                 * @param nsteps Steps of the advance.
                 */
                void collect_records(std::size_t nsteps);

                /**
                 * Queue the copies of the fields as they will be after the work queued so far.
                 *
                 * The fields go to the slot's device arrays on the stream, then to its page-locked
                 * host fields on the copy stream once those copies are done.
                 *
                 * @param slot Slot to copy into, its last snapshot handed over.
                 * @param steps Steps of the advance the snapshot is taken after.
                 */
                void queue_snapshot(SnapshotSlot &slot, std::size_t steps);

                /**
                 * Wait for a slot's copy to reach the host and hand it over.
                 *
                 * @param slot Slot to hand over.
                 * @param handler Receives it.
                 */
                static void deliver(const SnapshotSlot &slot, const SnapshotHandler &handler);

                /** How the kernels of a step learn which step they run. */
                enum class Launch {
                    /** They are given the step. */
                    Plain,

                    /** They read the step from the device counter and count it, so a graph can
                     * replay them. */
                    Replay
                };

                /** The sources, values and records of one step, as the kernels take them. */
                struct StepPoints
                {
                    /** Source cells on the device, the magnetic ones first. */
                    const gpu::DevicePoint *sources;

                    /** Sources that add to Hx or Hy. */
                    int magnetic;

                    /** Sources that add to Ez, after the magnetic ones. */
                    int electric;

                    /** Source values, pointed at the step's row unless step is set. */
                    const FloatType *values;

                    /** Probe records, pointed at the step's row unless step is set. */
                    FloatType *records;

                    /** Step counted on the device, for a graph to replay, or null. */
                    const int *step;
                };

                /**
                 * Run one step's kernels on the stream, on the device fields as they are then.
                 *
                 * H, the magnetic sources, E, the Ez sources and the probes, in that order: the
                 * level's update (launch_two_kernels() or launch_tiled()), then the Ez sources
                 * and the probes.
                 *
                 * @param coeffs View of the device coefficients.
                 * @param n Step of this advance, from zero; unused with Launch::Replay.
                 * @param magnetic Sources that add to Hx or Hy, first in the device sources.
                 * @param electric Sources that add to Ez, after the magnetic ones.
                 * @param launch How the kernels learn the step.
                 */
                void launch_step(const gpu::DeviceCoeffs &coeffs, std::size_t n,
                                 std::size_t magnetic, std::size_t electric, Launch launch);

                /**
                 * Level v1: queue an H kernel, the magnetic sources and an E kernel, one thread
                 * per cell.
                 *
                 * @param coeffs View of the device coefficients.
                 * @param points The step's sources, values and records.
                 * @return The device fields after the step.
                 */
                gpu::DeviceFields launch_two_kernels(const gpu::DeviceCoeffs &coeffs,
                                                     const StepPoints &points);

                /**
                 * Level v2: queue whole runs of GRAPH_STEPS steps as replays of one captured
                 * graph, captured again when its key changes.
                 *
                 * @param coeffs View of the device coefficients.
                 * @param from First step, from zero.
                 * @param to Step to stop before.
                 * @param magnetic Sources that add to Hx or Hy.
                 * @param electric Sources that add to Ez.
                 * @return The first step not queued, from plus a multiple of GRAPH_STEPS; from
                 *         itself when the run is shorter than a graph.
                 */
                std::size_t queue_graph(const gpu::DeviceCoeffs &coeffs, std::size_t from,
                                        std::size_t to, std::size_t magnetic, std::size_t electric);

                /**
                 * Level v3: build the coefficient table on the host, copy it and the index to
                 * the device, and find the tuples the interior shares.
                 *
                 * @param coeffs Coefficients on the backend's grid.
                 */
                void upload_table(const Coeffs &coeffs);

                /**
                 * Level v3: a view of the arrays with the table added: the interior's tuple when
                 * it shares one under the boundary in force, the table and index otherwise.
                 *
                 * @param arrays View of the eight arrays.
                 * @return The view.
                 */
                gpu::DeviceCoeffs table_coeffs(gpu::DeviceCoeffs arrays) const noexcept;

                /**
                 * Level v4: queue one kernel that updates H, adds the magnetic sources and
                 * updates E, each block keeping its tile's H in shared memory and writing the
                 * second set of arrays, which are then swapped with the first.
                 *
                 * @param coeffs View of the device coefficients.
                 * @param points The step's sources, values and records.
                 * @return The device fields after the step.
                 */
                gpu::DeviceFields launch_tiled(const gpu::DeviceCoeffs &coeffs,
                                               const StepPoints &points);

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

                    /** Device Ez, which of the two arrays the tiled kernel swaps between. */
                    const void *ez = nullptr;

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
                CudaOptions m_options;

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

                /** Device Ez the tiled kernel of v4 writes a step to, swapped with m_ez after it.
                 */
                gpu::DeviceBuffer<FloatType> m_ez_next;

                /** Device Hx the tiled kernel writes a step to, swapped with m_hx after it. */
                gpu::DeviceBuffer<FloatType> m_hx_next;

                /** Device Hy the tiled kernel writes a step to, swapped with m_hy after it. */
                gpu::DeviceBuffer<FloatType> m_hy_next;

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

                /** Table entries on the device, from v3. */
                gpu::DeviceBuffer<gpu::CoeffTuple> m_table_entries;

                /** Entry of each cell on the device, laid out as the fields, from v3. */
                gpu::DeviceBuffer<std::uint8_t> m_table_index;

                /** Tuple every interior cell shares with the PML in force, if one is shared. */
                std::optional<gpu::CoeffTuple> m_uniform_pml;

                /** Tuple every interior cell shares with no PML in force, if one is shared. */
                std::optional<gpu::CoeffTuple> m_uniform_plain;

                /** Graph of GRAPH_STEPS steps, captured by the first advance that needs it, from
                 * v2. */
                gpu::Graph m_graph;

                /** What m_graph was captured with, captured again when this changes. */
                GraphKey m_graph_key;

                /** Copy stream and slots of async_snapshots, or null without it. */
                std::unique_ptr<AsyncSnapshots> m_async;
            };
        }
    }
}

#endif
