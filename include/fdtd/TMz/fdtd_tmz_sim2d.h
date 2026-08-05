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

#ifndef FDTD_TMZ_SIM2D_H
#define FDTD_TMZ_SIM2D_H

#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_fields2d.h"
#include "fdtd/TMz/fdtd_tmz_grid2d.h"
#include "fdtd/TMz/fdtd_tmz_material2d.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <memory>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /**
         * A 2D TMz simulation: grid, material, fields and the backend that steps them.
         *
         * The material starts as vacuum and the fields at zero. The coefficients for the
         * material and time step are uploaded to the backend on construction.
         */
        class Sim2D
        {
        public:
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

                /** Backend to run on. */
                BackendType backend = BackendType::Serial;

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
            Sim2D(std::size_t nx, std::size_t ny);

            /**
             * Create a simulation.
             *
             * @param nx Cells along x, at least 1.
             * @param ny Cells along y, at least 1.
             * @param options Settings of the simulation.
             * @throws std::invalid_argument On a bad grid, a Courant number outside (0, 1],
             *         a negative or non-finite dt, or, with check_stability, an explicit
             *         dt above the Courant limit.
             * @throws std::runtime_error When the backend is not compiled in.
             */
            Sim2D(std::size_t nx, std::size_t ny, const Options &options);

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
            const Grid2D &grid() const noexcept;

            /**
             * Time step in use.
             *
             * @return Time step in seconds.
             */
            double dt() const noexcept;

            /**
             * Material of the simulation.
             *
             * @return The material.
             */
            const Material2D &material() const noexcept;

            /**
             * Backend the simulation runs on.
             *
             * @return The backend.
             */
            const Backend &backend() const noexcept;

        private:
            /** Settings the simulation was created with. */
            Options m_options;

            /** Grid of the simulation. */
            Grid2D m_grid;

            /** Time step in seconds. */
            double m_dt;

            /** Per-cell material. */
            Material2D m_material;

            /** Host copy of the fields. */
            Fields2D m_fields;

            /** Backend that owns the working fields and steps them. */
            std::unique_ptr<Backend> m_backend;
        };
    }
}

#endif
