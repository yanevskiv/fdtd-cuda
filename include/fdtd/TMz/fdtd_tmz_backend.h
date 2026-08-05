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
#include "fdtd/TMz/fdtd_tmz_coeffs2d.h"
#include "fdtd/TMz/fdtd_tmz_fields2d.h"
#include "fdtd/TMz/fdtd_tmz_grid2d.h"

#include <cstddef>
#include <memory>
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

        /**
         * A backend that owns a copy of the fields and advances them in time.
         *
         * Fields move in with upload() and out with download(); step() works on the
         * backend's own copy, which may live on another device. The coefficients move in
         * with upload() too, and must be there before the first step. The outer boundary
         * is PEC until set_boundary() says otherwise.
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
            virtual const Grid2D &grid() const noexcept = 0;

            /**
             * Copy fields into the backend.
             *
             * @param fields Fields on the backend's grid.
             * @throws std::invalid_argument When the grids differ.
             */
            virtual void upload(const Fields2D &fields) = 0;

            /**
             * Copy the backend's fields out.
             *
             * @param fields Destination on the backend's grid.
             * @throws std::invalid_argument When the grids differ.
             */
            virtual void download(Fields2D &fields) const = 0;

            /**
             * Copy update coefficients into the backend, replacing any before.
             *
             * @param coeffs Coefficients on the backend's grid.
             * @throws std::invalid_argument When the grids differ.
             */
            virtual void upload(const Coeffs2D &coeffs) = 0;

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
             * Advance the fields: per step H, then Ez inside, then the edge Ez.
             *
             * @param nsteps Number of time steps.
             * @throws std::logic_error When no coefficients were uploaded.
             */
            virtual void step(std::size_t nsteps) = 0;

        protected:
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
        };

        /**
         * Short lowercase name of a backend type.
         *
         * @param type Backend type.
         * @return "serial", "openmp" or "cuda".
         */
        std::string_view backend_name(BackendType type) noexcept;

        /**
         * Backends compiled into this build, serial first.
         *
         * @return The available types.
         */
        std::vector<BackendType> available_backends();

        /**
         * Whether a backend is compiled into this build.
         *
         * @param type Backend type.
         * @return True when make_backend() can create it.
         */
        bool is_backend_available(BackendType type);

        /**
         * Create a backend with zeroed fields.
         *
         * @param type Backend type.
         * @param grid Grid the backend works on.
         * @return The new backend.
         * @throws std::runtime_error When the backend is not compiled in.
         */
        std::unique_ptr<Backend> make_backend(BackendType type, const Grid2D &grid);
    }
}

#endif
