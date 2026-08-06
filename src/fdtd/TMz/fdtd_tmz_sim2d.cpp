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

// Standard library for algorithms, math, errors and moves.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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

    // Check the grid, keep the material and upload its coefficients.
    void Sim2D::set_material(const Material2D &material)
    {
        const Grid2D &grid = material.grid();
        if (grid.nx() != m_grid.nx() || grid.ny() != m_grid.ny() || grid.dx() != m_grid.dx() ||
            grid.dy() != m_grid.dy()) {
            throw std::invalid_argument("Sim2D: material grid does not match");
        }
        m_material = material;
        m_backend->upload(Coeffs2D(m_material, m_dt));
    }

    // Return the backend.
    const Backend &Sim2D::backend() const noexcept
    {
        return *m_backend;
    }

    // Check the cell and keep the source.
    void Sim2D::add_source(std::size_t i, std::size_t j, Component component, Waveform waveform)
    {
        const std::size_t nx = m_grid.nx();
        const std::size_t ny = m_grid.ny();
        if (i >= nx || j >= ny) {
            throw std::invalid_argument("Sim2D: source outside the grid");
        }
        if (!waveform) {
            throw std::invalid_argument("Sim2D: empty source waveform");
        }
        const bool edge = i == 0 || j == 0 || i + 1 == nx || j + 1 == ny;
        if ((component == Component::Ez && edge && m_options.boundary == Boundary::Pec) ||
            (component == Component::Hx && j + 1 == ny) ||
            (component == Component::Hy && i + 1 == nx)) {
            throw std::invalid_argument("Sim2D: source on a cell that is never updated");
        }
        m_sources.push_back({{component, m_grid.index(i, j)}, std::move(waveform)});
    }

    // Return the number of sources.
    std::size_t Sim2D::source_count() const noexcept
    {
        return m_sources.size();
    }

    // Merge sources on the same cell, sample every source for the steps ahead, then step
    // the backend and count the steps.
    void Sim2D::run(std::size_t nsteps)
    {
        std::vector<PointInjection> points;
        std::vector<std::size_t> slots;
        slots.reserve(m_sources.size());
        for (const Source &source : m_sources) {
            const auto it =
                std::find_if(points.begin(), points.end(), [&source](const PointInjection &point) {
                    return point.component == source.point.component &&
                           point.index == source.point.index;
                });
            slots.push_back(static_cast<std::size_t>(it - points.begin()));
            if (it == points.end()) {
                points.push_back(source.point);
            }
        }

        const std::size_t npoints = points.size();
        std::vector<FloatType> values(nsteps * npoints, FloatType(0));
        for (std::size_t s = 0; s < m_sources.size(); ++s) {
            const Source &source = m_sources[s];
            const double offset = source.point.component == Component::Ez ? 1.0 : 0.5;
            const auto samples =
                sample_waveform(source.waveform, m_dt, m_step_count, nsteps, offset);
            for (std::size_t n = 0; n < nsteps; ++n) {
                values[n * npoints + slots[s]] += samples[n];
            }
        }

        m_backend->step(nsteps, points, values);
        m_step_count += nsteps;
    }

    // Return the step count.
    std::size_t Sim2D::step_count() const noexcept
    {
        return m_step_count;
    }

    // Return the simulated time.
    double Sim2D::time() const noexcept
    {
        return static_cast<double>(m_step_count) * m_dt;
    }

    // Return the host copy of the fields.
    const Fields2D &Sim2D::fields() const noexcept
    {
        return m_fields;
    }

    // Copy the fields from the backend.
    void Sim2D::download()
    {
        m_backend->download(m_fields);
    }
}
