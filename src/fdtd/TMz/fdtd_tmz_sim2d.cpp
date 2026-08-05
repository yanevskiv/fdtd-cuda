/*
 * C++ file for the 2D TMz simulation.
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
#include "fdtd/TMz/fdtd_tmz_sim2d.h"

// Coefficients and the Courant limit.
#include "fdtd/TMz/fdtd_tmz_coeffs2d.h"
#include "fdtd/TMz/fdtd_tmz_courant.h"

// Standard library for math and errors.
#include <cmath>
#include <stdexcept>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Helpers private to this file.
    namespace
    {
        // Return the explicit dt if it is valid, otherwise derive it from the Courant
        // number; reject an explicit dt above the limit when stability is checked.
        double options_dt(const Grid2D &grid, const Sim2D::Options &options)
        {
            if (options.dt == 0) {
                return courant_dt(grid, options.courant);
            }
            if (!(options.dt > 0) || !std::isfinite(options.dt)) {
                throw std::invalid_argument("Sim2D: dt must be positive and finite, or zero");
            }
            if (options.check_stability && options.dt > courant_dt(grid, 1.0)) {
                throw std::invalid_argument("Sim2D: dt is above the Courant limit");
            }
            return options.dt;
        }
    }

    // Delegate to the full constructor with the default options.
    Sim2D::Sim2D(std::size_t nx, std::size_t ny)
        : Sim2D(nx, ny, Options{})
    {
        // Empty.
    }

    // Build the grid, the vacuum material and the backend, and upload the coefficients.
    Sim2D::Sim2D(std::size_t nx, std::size_t ny, const Options &options)
        : m_options(options)
        , m_grid(nx, ny, options.dx, options.dy)
        , m_dt(options_dt(m_grid, options))
        , m_material(m_grid)
        , m_fields(m_grid)
        , m_backend(make_backend(options.backend, m_grid))
    {
        m_backend->upload(Coeffs2D(m_material, m_dt));
        m_backend->set_boundary(options.boundary);
        m_backend->upload(m_fields);
    }

    // Return the options.
    const Sim2D::Options &Sim2D::options() const noexcept
    {
        return m_options;
    }

    // Return the grid.
    const Grid2D &Sim2D::grid() const noexcept
    {
        return m_grid;
    }

    // Return the time step.
    double Sim2D::dt() const noexcept
    {
        return m_dt;
    }

    // Return the material.
    const Material2D &Sim2D::material() const noexcept
    {
        return m_material;
    }

    // Return the backend.
    const Backend &Sim2D::backend() const noexcept
    {
        return *m_backend;
    }
}
