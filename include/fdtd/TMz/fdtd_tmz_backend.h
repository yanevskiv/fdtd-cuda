/*
 * C++ header for the 2D TMz backend interface and registry.
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

#ifndef FDTD_TMZ_BACKEND_H
#define FDTD_TMZ_BACKEND_H

#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_component.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** The backends a simulation can run on. */
        enum class BackendType {
            /** Single-threaded reference backend, the oracle for the others. */
            Serial,

            /** Multi-threaded CPU backend. */
            OpenMP,

            /** GPU backend. */
            Cuda
        };

        /** A cell of one field component that a soft source adds to every step. */
        struct PointInjection
        {
            /** Component the value is added to. */
            Component component;

            /** Flat index of the cell, as given by Grid::index. */
            std::size_t index;
        };

        /** A cell of one field component that a probe records at the end of every step. */
        struct PointProbe
        {
            /** Component the value is read from. */
            Component component;

            /** Flat index of the cell, as given by Grid::index. */
            std::size_t index;
        };

        /**
         * Modelled memory traffic and arithmetic of one time step.
         *
         * The model counts each array element a kernel touches as read once and written once per pass, with the neighbours of a cell coming
         * from cache, so it is the traffic to main memory a kernel cannot avoid. Sources and probes are left out.
         */
        struct StepTraffic
        {
            /** Bytes read from and written to memory. */
            double bytes = 0;

            /** Floating-point operations, a multiply-add counted as two. */
            double flops = 0;
        };

        /** Where the kernels of a traffic model take their coefficients from. */
        struct TrafficModel
        {
            /** Whether the core cells, the interior less its last column and row, read a table of a few tuples instead of the arrays. */
            bool table = false;

            /** Whether every cell reads a one-byte table index in each pass. */
            bool index = false;

            /** Whether H and E are updated in one pass, each field read and written once. */
            bool one_pass = false;
        };

        /**
         * Traffic of one step of the kernels with separate H and E passes, the serial backend's.
         *
         * The H pass reads Ez once per cell, and per Hx or Hy update reads the field and its two coefficients and writes the field: 4 words
         * and 5 flops. The E pass takes per interior cell 6 words (Hx, Hy, Ez and two coefficients read, Ez written) and 8 flops; per PML
         * cell 11 words (Hx, Hy, Ezx, Ezy and four coefficients read, Ezx, Ezy and Ez written) and 11 flops; per edge cell 3 words written
         * under PEC and PML, and the interior's count under PMC. With a table, the core cells read no coefficients, in either pass; with an
         * index, every cell reads one byte in each pass.
         *
         * In one pass, a cell reads Ez, Hx and Hy once and writes them once (6 words), its H coefficients as in two passes, and its E
         * coefficients: 2 words in the interior, 4 and Ezx and Ezy read and written in the PML (8), Ezx and Ezy written on the edge under
         * PEC and PML or 2 coefficients under PMC (2). The halo a tile computes again is left out, and the flops are those of two passes.
         *
         * @param grid Grid of the fields.
         * @param pml Thickness of the active PML in cells, zero for none.
         * @param boundary Outer boundary condition.
         * @param model Where the coefficients come from; the arrays by default.
         * @return The traffic, a word being sizeof(FloatType) bytes.
         */
        StepTraffic two_pass_traffic(const Grid &grid, std::size_t pml, Boundary boundary, const TrafficModel &model = {});

        /**
         * A backend that owns a copy of the fields and advances them in time.
         *
         * Fields move in with upload() and out with download(); step() works on the backend's own copy, which may live on another device.
         * The coefficients move in with upload() too, and must be there before the first step. The outer boundary is PEC until
         * set_boundary() says otherwise.
         */
        class Backend
        {
        public:
            /** Release the backend's resources. */
            virtual ~Backend();

            /**
             * Short lowercase name of the backend.
             *
             * @return "serial", "openmp" or "cuda".
             */
            virtual std::string_view name() const noexcept = 0;

            /**
             * Type of the backend.
             *
             * @return The backend type.
             */
            virtual BackendType type() const noexcept = 0;

            /**
             * Grid the backend was made for.
             *
             * @return The grid.
             */
            virtual const Grid &grid() const noexcept = 0;

            /**
             * Copy fields into the backend, Ezx and Ezy included.
             *
             * @param fields Fields on the backend's grid.
             * @throws std::invalid_argument When the grids differ.
             */
            virtual void upload(const Fields &fields) = 0;

            /**
             * Copy the backend's fields out, Ezx and Ezy included.
             *
             * @param fields Destination on the backend's grid.
             * @throws std::invalid_argument When the grids differ.
             */
            virtual void download(Fields &fields) const = 0;

            /**
             * Copy update coefficients into the backend, replacing any before.
             *
             * @param coeffs Coefficients on the backend's grid.
             * @throws std::invalid_argument When the grids differ.
             */
            virtual void upload(const Coeffs &coeffs) = 0;

            /**
             * Set the outer boundary condition for the steps that follow.
             *
             * @param boundary Outer boundary condition.
             */
            virtual void set_boundary(Boundary boundary) = 0;

            /**
             * Outer boundary condition.
             *
             * @return The boundary.
             */
            virtual Boundary boundary() const noexcept = 0;

            /**
             * Advance the fields with no sources.
             *
             * @param nsteps Number of time steps.
             * @throws std::logic_error When no coefficients were uploaded.
             */
            void step(std::size_t nsteps);

            /**
             * Advance the fields and add soft point sources.
             *
             * Per step: H, then the Hx and Hy sources, then Ez inside, then the edge Ez, then the Ez sources. Step n adds values[n *
             * points.size() + k] to point k.
             *
             * @param nsteps Number of time steps.
             * @param points Distinct cells to add to.
             * @param values One value per point per step, step-major.
             * @throws std::invalid_argument When the values do not match the points and steps, a point lies outside the grid, or two points
             *         are the same.
             * @throws std::logic_error When no coefficients were uploaded.
             */
            void step(std::size_t nsteps, std::span<const PointInjection> points, std::span<const FloatType> values);

            /**
             * Receives a snapshot taken during a step() call: the steps of the call run before it, and the fields then. The fields are
             * valid only during the call.
             */
            using SnapshotHandler = std::function<void(std::size_t steps, const Fields &fields)>;

            /**
             * Advance the fields, add soft point sources, and hand over the fields after given steps of the call.
             *
             * As step() with sources, and after each step count in snapshots the handler receives the fields, in order. A backend may copy
             * the fields out while it goes on stepping, and call the handler while the steps after the snapshot run.
             *
             * @param nsteps Number of time steps.
             * @param points Distinct cells to add to.
             * @param values One value per point per step, step-major.
             * @param snapshots Step counts of the call to take the fields after, increasing, each from 1 to nsteps.
             * @param handler Receives each snapshot.
             * @throws std::invalid_argument As step(), or when the snapshot steps are out of order or range.
             * @throws std::logic_error When no coefficients were uploaded.
             */
            void step(std::size_t nsteps, std::span<const PointInjection> points, std::span<const FloatType> values,
                      std::span<const std::size_t> snapshots, const SnapshotHandler &handler);

            /**
             * Replace the probes and drop everything recorded so far.
             *
             * Every step after this appends the value of each probe at the end of the step, after the sources: Ez is then at (n + 1) * dt
             * and H at (n + 1/2) * dt. The records stay in the backend until fetch_probes() copies them out.
             *
             * @param points Cells to record; the same cell may appear twice.
             * @throws std::invalid_argument When a point lies outside the grid.
             */
            void set_probes(std::span<const PointProbe> points);

            /**
             * Number of probes set.
             *
             * @return The probe count.
             */
            virtual std::size_t probe_count() const noexcept = 0;

            /**
             * Number of steps recorded since the probes were set.
             *
             * @return The recorded step count.
             */
            virtual std::size_t probe_steps() const noexcept = 0;

            /**
             * Copy the probe records out.
             *
             * @return probe_steps() * probe_count() values, step-major: step n of probe k is at n * probe_count() + k.
             */
            virtual std::vector<FloatType> fetch_probes() const = 0;

            /**
             * Modelled traffic of one step with the coefficients and boundary set now.
             *
             * @return The traffic of the backend's kernels.
             */
            virtual StepTraffic step_traffic() const noexcept = 0;

        protected:
            /**
             * Replace the probes and clear the records; called by set_probes() after its checks.
             *
             * @param points Cells inside the grid.
             */
            virtual void reset_probes(std::span<const PointProbe> points) = 0;

            /**
             * Advance the fields and add the sources; called by step() after its checks.
             *
             * @param nsteps Number of time steps.
             * @param points Distinct cells inside the grid.
             * @param values One value per point per step, step-major.
             * @throws std::logic_error When no coefficients were uploaded.
             */
            virtual void advance(std::size_t nsteps, std::span<const PointInjection> points, std::span<const FloatType> values) = 0;

            /**
             * Advance the fields, add the sources and hand over the snapshots; called by step() after its checks.
             *
             * The default advances to each snapshot, downloads the fields into a copy of its own and hands them over before it goes on.
             * With no snapshots it advances alone, with no copy to make.
             *
             * @param nsteps Number of time steps.
             * @param points Distinct cells inside the grid.
             * @param values One value per point per step, step-major.
             * @param snapshots Increasing step counts from 1 to nsteps.
             * @param handler Receives each snapshot.
             * @throws std::logic_error When no coefficients were uploaded.
             */
            virtual void advance_snapshots(std::size_t nsteps, std::span<const PointInjection> points, std::span<const FloatType> values,
                                           std::span<const std::size_t> snapshots, const SnapshotHandler &handler);

            /** Create a backend. */
            Backend() = default;

            /** Copy a backend; protected so only derived classes copy, never through a base. */
            Backend(const Backend &) = default;

            /** Move a backend; protected so only derived classes move, never through a base. */
            Backend(Backend &&) = default;

            /**
             * Copy-assign a backend; protected so only derived classes copy.
             *
             * @return This backend.
             */
            Backend &operator=(const Backend &) = default;

            /**
             * Move-assign a backend; protected so only derived classes move.
             *
             * @return This backend.
             */
            Backend &operator=(Backend &&) = default;

        private:
            /**
             * Check the sources of a step() call.
             *
             * @param nsteps Number of time steps.
             * @param points Cells to add to.
             * @param values Values to add.
             * @throws std::invalid_argument When the values do not match the points and steps, a point lies outside the grid, or two points
             *         are the same.
             */
            void check_sources(std::size_t nsteps, std::span<const PointInjection> points, std::span<const FloatType> values) const;
        };

        /**
         * Short lowercase name of a backend type.
         *
         * @param type Backend type.
         * @return "serial", "openmp" or "cuda".
         */
        std::string_view backend_name(BackendType type) noexcept;

        /**
         * Backends compiled into this build, serial first; CUDA only when a device is present.
         *
         * @return The available types.
         */
        std::vector<BackendType> available_backends();

        /**
         * Whether a backend is compiled into this build and, for CUDA, a device is present.
         *
         * @param type Backend type.
         * @return True when make_backend() can create it.
         */
        bool is_backend_available(BackendType type);

        /**
         * Canonical name of a variant of a backend, which the backend reads back.
         *
         * Each backend reads its own grammar, whether it is compiled in or not: serial reads only `default`, and OpenMP and CUDA read the
         * names their settings describe. `default` names a backend's default settings.
         *
         * @param type Backend type.
         * @param variant Variant name, as `v2` for OpenMP.
         * @return The canonical name: `default` for serial, every setting written for the others.
         * @throws std::invalid_argument When the backend does not read the name, with the reason and the name in the message.
         */
        std::string variant_name(BackendType type, std::string_view variant);

        /**
         * Whether a backend reads a variant name.
         *
         * @param type Backend type.
         * @param variant Variant name.
         * @return True when variant_name() takes the name for the backend.
         */
        bool reads_variant(BackendType type, std::string_view variant);
    }
}

#endif
