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
        enum class BackendKind {
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
         * backend's own copy, which may live on another device.
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
             * Kind of the backend.
             *
             * @return The backend kind.
             */
            virtual BackendKind kind() const noexcept = 0;

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
             * Advance the fields.
             *
             * @param nsteps Number of time steps.
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
         * Short lowercase name of a backend kind.
         *
         * @param kind Backend kind.
         * @return "serial", "openmp" or "cuda".
         */
        std::string_view backend_name(BackendKind kind) noexcept;

        /**
         * Backends compiled into this build, serial first.
         *
         * @return The available kinds.
         */
        std::vector<BackendKind> available_backends();

        /**
         * Whether a backend is compiled into this build.
         *
         * @param kind Backend kind.
         * @return True when make_backend() can create it.
         */
        bool is_backend_available(BackendKind kind);

        /**
         * Create a backend with zeroed fields.
         *
         * @param kind Backend kind.
         * @param grid Grid the backend works on.
         * @return The new backend.
         * @throws std::runtime_error When the backend is not compiled in.
         */
        std::unique_ptr<Backend> make_backend(BackendKind kind, const Grid2D &grid);
    }
}

#endif
