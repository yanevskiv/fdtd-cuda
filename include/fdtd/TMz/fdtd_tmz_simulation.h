/*
 * C++ header for the 2D TMz simulation.
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

#ifndef FDTD_TMZ_SIMULATION_H
#define FDTD_TMZ_SIMULATION_H

#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_component.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/TMz/fdtd_tmz_material.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <memory>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** How the OpenMP backend shares the cells of a phase among its threads. */
        enum class OmpSchedule {
            /** A static `omp for` over the rows, the baseline. The default. */
            Rows,

            /** One contiguous range of cells in row-major order per thread. */
            Collapse2,

            /** A static `omp for` over tiles of omp_tile_x by omp_tile_y cells. */
            Tiles,

            /** One task per tile per phase, ordered by dependences on the neighbouring tiles. */
            Tasks
        };

        /** Where the OpenMP backend opens its parallel regions. */
        enum class OmpRegion {
            /** One region around the whole run, with the phases as work-sharing loops. */
            Run,

            /** One region per phase of every step. */
            Phase
        };

        /** How the CUDA backend lays the update of a step out in kernels. */
        enum class CudaKernel {
            /** One E kernel for every cell, each thread taking its cell's edge, PML or interior
             * branch. The default. */
            Fused,

            /** Three E kernels with no branch on the cell: the interior, the PML slabs and the
             * edge. */
            Split
        };

        /**
         * A 2D TMz simulation: grid, material, fields and the backend that steps them.
         *
         * The material starts as vacuum and the fields at zero. The coefficients for the
         * material and time step are uploaded to the backend on construction.
         */
        class Simulation
        {
        public:
            /** Copy of the fields taken after a number of steps. */
            struct Snapshot
            {
                /** Steps run when the copy was taken; Ez is at step * dt, H half a step earlier. */
                std::size_t step;

                /** The fields at that step. */
                Fields fields;
            };

            /** Settings of a simulation, each with a usable default. */
            struct Options
            {
                /** Cell size along x in meters, positive. */
                FloatType dx = FloatType(1e-3);

                /** Cell size along y in meters, positive. */
                FloatType dy = FloatType(1e-3);

                /** Courant number S in (0, 1], used when dt is zero. */
                double courant = 0.99;

                /** Explicit time step in seconds, or zero to derive it from courant. */
                double dt = 0;

                /** Outer boundary condition. */
                Boundary boundary = Boundary::Pec;

                /** PML thickness in cells, used with Boundary::Pml; zero turns the PML off. */
                std::size_t pml_thickness = 20;

                /** Grading exponent m of the PML conductivity, positive. */
                double pml_order = 3;

                /** Target PML reflection R0 at normal incidence, in (0, 1). */
                double pml_reflection = 1e-6;

                /** Backend to run on. */
                BackendType backend = BackendType::Serial;

                /** How the OpenMP backend shares the cells among its threads. */
                OmpSchedule omp_schedule = OmpSchedule::Rows;

                /** Thread count of the OpenMP backend, 0 for the OpenMP default. */
                std::size_t omp_threads = 0;

                /** Tile width in cells for the Tiles and Tasks schedules, positive. */
                std::size_t omp_tile_x = 128;

                /** Tile height in cells for the Tiles and Tasks schedules, positive. */
                std::size_t omp_tile_y = 32;

                /** Whether the OpenMP backend's inner loops carry `omp simd`; on by default,
                 * the fastest on the M5 sweep. */
                bool omp_simd = true;

                /** Where the OpenMP backend opens its parallel regions; Tasks takes Run only. */
                OmpRegion omp_region = OmpRegion::Run;

                /** How the CUDA backend lays a step out in kernels. */
                CudaKernel cuda_kernel = CudaKernel::Fused;

                /** Threads along x of a CUDA thread block, positive. */
                std::size_t cuda_block_x = 32;

                /** Threads along y of a CUDA thread block, positive, with cuda_block_x *
                 * cuda_block_y within the device's threads per block. */
                std::size_t cuda_block_y = 8;

                /** Whether an explicit dt above the Courant limit is rejected. */
                bool check_stability = true;
            };

            /**
             * Create a simulation with the default options.
             *
             * @param nx Cells along x, at least 1.
             * @param ny Cells along y, at least 1.
             * @throws std::invalid_argument On a bad grid.
             */
            Simulation(std::size_t nx, std::size_t ny);

            /**
             * Create a simulation.
             *
             * @param nx Cells along x, at least 1.
             * @param ny Cells along y, at least 1.
             * @param options Settings of the simulation.
             * @throws std::invalid_argument On a bad grid, a Courant number outside (0, 1],
             *         a negative or non-finite dt, or, with check_stability, an explicit
             *         dt above the Courant limit, or, with an active PML, PML settings that
             *         Material::set_pml() rejects, or backend settings the backend rejects.
             * @throws std::runtime_error When the backend is not compiled in, or is CUDA and no
             *         device is present.
             * @throws CudaError When a CUDA call of the backend fails.
             */
            Simulation(std::size_t nx, std::size_t ny, const Options &options);

            /**
             * Settings the simulation was created with.
             *
             * @return The options.
             */
            const Options &options() const noexcept;

            /**
             * Grid of the simulation.
             *
             * @return The grid.
             */
            const Grid &grid() const noexcept;

            /**
             * Time step in use.
             *
             * @return Time step in seconds.
             */
            double dt() const noexcept;

            /**
             * Material of the simulation, with the PML of the options graded in.
             *
             * @return The material.
             */
            const Material &material() const noexcept;

            /**
             * Replace the material and upload the coefficients rebuilt from it.
             *
             * The PML of the options replaces any the material carries, so the PML always
             * follows the boundary.
             *
             * @param material New material, on a grid with the same size and cells.
             * @throws std::invalid_argument When the material's grid does not match.
             */
            void set_material(const Material &material);

            /**
             * Thickness of the active PML.
             *
             * @return pml_thickness with Boundary::Pml, zero with any other boundary.
             */
            std::size_t pml_thickness() const noexcept;

            /**
             * Backend the simulation runs on.
             *
             * @return The backend.
             */
            const Backend &backend() const noexcept;

            /**
             * Add a soft point source: the waveform is added to one cell every step.
             *
             * Ez sources are sampled at the end of each step, (n + 1) * dt, and added after
             * the edge update. Hx and Hy sources are sampled at (n + 1/2) * dt and added
             * after the H update. Sources on the same cell add up.
             *
             * @param i Column of the cell.
             * @param j Row of the cell.
             * @param component Field the waveform is added to.
             * @param waveform Value at a time in seconds.
             * @throws std::invalid_argument When the cell is outside the grid, is never
             *         updated (Ez on the edge under PEC or PML, Hx in the last row, Hy in the last
             *         column) or lies in the PML (within pml_thickness() cells of the edge, for
             *         any component), or the waveform is empty.
             */
            void add_source(std::size_t i, std::size_t j, Component component, Waveform waveform);

            /**
             * Number of sources added.
             *
             * @return The source count.
             */
            std::size_t source_count() const noexcept;

            /**
             * Add a point probe that records one component of a cell every step.
             *
             * Each step records the value at its end: Ez at (n + 1) * dt and Hx or Hy at
             * (n + 1/2) * dt, as probe_times() gives them. Probes are added before the
             * first run so that they all cover the same steps.
             *
             * @param i Column of the cell.
             * @param j Row of the cell.
             * @param component Field to record.
             * @return Id of the probe, counting from zero in the order added.
             * @throws std::invalid_argument When the cell is outside the grid.
             * @throws std::logic_error When steps have already run.
             */
            std::size_t add_probe(std::size_t i, std::size_t j, Component component);

            /**
             * Number of probes added.
             *
             * @return The probe count.
             */
            std::size_t probe_count() const noexcept;

            /**
             * Values a probe recorded, fetched from the backend.
             *
             * @param id Id from add_probe().
             * @return One value per step run.
             * @throws std::out_of_range When the id is unknown.
             */
            std::vector<FloatType> probe_data(std::size_t id) const;

            /**
             * Times a probe's values belong to.
             *
             * @param id Id from add_probe().
             * @return (n + 1) * dt for Ez, (n + 1/2) * dt for Hx and Hy, one per step run.
             * @throws std::out_of_range When the id is unknown.
             */
            std::vector<double> probe_times(std::size_t id) const;

            /**
             * Schedule a snapshot of Ez, Hx and Hy after a number of steps.
             *
             * A later run() stops at the step to copy the fields, then carries on. A step
             * equal to step_count() is taken at once, and a step already scheduled or taken
             * is ignored.
             *
             * @param step Steps run when the snapshot is taken.
             * @throws std::invalid_argument When the step is below step_count().
             */
            void add_snapshot(std::size_t step);

            /**
             * Take a snapshot of the fields as they are now, without keeping it.
             *
             * @return The fields at step_count().
             */
            Snapshot snapshot() const;

            /**
             * Snapshots taken so far.
             *
             * @return The snapshots, in step order.
             */
            const std::vector<Snapshot> &snapshots() const noexcept;

            /**
             * Advance the simulation, stopping at scheduled snapshots.
             *
             * @param nsteps Number of time steps to run.
             */
            void run(std::size_t nsteps);

            /**
             * Number of time steps run so far.
             *
             * @return The step count.
             */
            std::size_t step_count() const noexcept;

            /**
             * Simulated time so far.
             *
             * @return step_count() * dt() in seconds.
             */
            double time() const noexcept;

            /**
             * Host copy of the fields, as of the last download().
             *
             * @return The fields.
             */
            const Fields &fields() const noexcept;

            /** Copy the fields from the backend into the host copy. */
            void download();

            /**
             * Modelled memory traffic and arithmetic of one time step on the backend.
             *
             * @return The backend's step_traffic().
             */
            StepTraffic step_traffic() const noexcept;

        private:
            /**
             * Sample the sources and step the backend.
             *
             * @param nsteps Number of time steps to run.
             */
            void advance(std::size_t nsteps);

            /** A point source: the cell it adds to and its waveform. */
            struct Source
            {
                /** Component and cell. */
                PointInjection point;

                /** Value at a time in seconds. */
                Waveform waveform;
            };

            /** Settings the simulation was created with. */
            Options m_options;

            /** Grid of the simulation. */
            Grid m_grid;

            /** Time step in seconds. */
            double m_dt;

            /** Per-cell material, holding the PML profiles. */
            Material m_material;

            /** Host copy of the fields. */
            Fields m_fields;

            /** Backend that owns the working fields and steps them. */
            std::unique_ptr<Backend> m_backend;

            /** Time steps run so far. */
            std::size_t m_step_count = 0;

            /** Point sources, in the order added. */
            std::vector<Source> m_sources;

            /** Probe cells, in the order added. */
            std::vector<PointProbe> m_probes;

            /** Steps of the snapshots still to take, in increasing order. */
            std::vector<std::size_t> m_snapshot_steps;

            /** Snapshots taken, in step order. */
            std::vector<Snapshot> m_snapshots;
        };

        /**
         * Create a backend with zeroed fields and the default settings.
         *
         * @param type Backend type.
         * @param grid Grid the backend works on.
         * @return The new backend.
         * @throws std::runtime_error When the backend is not compiled in, or is CUDA and no
         *         device is present.
         */
        std::unique_ptr<Backend> make_backend(BackendType type, const Grid &grid);

        /**
         * Create a backend with zeroed fields and the settings of a simulation.
         *
         * @param type Backend type.
         * @param grid Grid the backend works on.
         * @param options Settings; a backend reads only its own, as the omp_ or cuda_ fields.
         * @return The new backend.
         * @throws std::runtime_error When the backend is not compiled in, or is CUDA and no
         *         device is present.
         * @throws std::invalid_argument When the backend's settings are invalid.
         */
        std::unique_ptr<Backend> make_backend(BackendType type, const Grid &grid,
                                              const Simulation::Options &options);
    }
}

#endif
