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
#include "fdtd/TMz/fdtd_tmz_grid2d.h"

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
        // Return a valid explicit dt within the limit, or derive dt from the Courant number.
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

        // Grade the material's PML from the options, or clear it unless the boundary is Pml.
        void apply_pml(Material2D &material, const Sim2D::Options &options)
        {
            const std::size_t thickness =
                options.boundary == Boundary::Pml ? options.pml_thickness : 0;
            material.set_pml(thickness, options.pml_order, options.pml_reflection);
        }
    }

    // Delegate to the full constructor with the default options.
    Sim2D::Sim2D(std::size_t nx, std::size_t ny)
        : Sim2D(nx, ny, Options{})
    {
        // Empty.
    }

    // Build the grid, the vacuum material with its PML and the backend, and upload coefficients.
    Sim2D::Sim2D(std::size_t nx, std::size_t ny, const Options &options)
        : m_options(options)
        , m_grid(nx, ny, options.dx, options.dy)
        , m_dt(options_dt(m_grid, options))
        , m_material(m_grid)
        , m_fields(m_grid)
        , m_backend(make_backend(options.backend, m_grid))
    {
        apply_pml(m_material, m_options);
        m_backend->upload(Coeffs2D(m_material, m_dt));
        m_backend->set_boundary(options.boundary);
        m_backend->upload(m_fields);
    }

    // Return the active PML thickness.
    std::size_t Sim2D::pml_thickness() const noexcept
    {
        return m_material.pml_thickness();
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

    // Check the grid, keep the material with the options' PML and upload its coefficients.
    void Sim2D::set_material(const Material2D &material)
    {
        const Grid2D &grid = material.grid();
        if (grid.nx() != m_grid.nx() || grid.ny() != m_grid.ny() || grid.dx() != m_grid.dx() ||
            grid.dy() != m_grid.dy()) {
            throw std::invalid_argument("Sim2D: material grid does not match");
        }
        Material2D next = material;
        apply_pml(next, m_options);
        m_material = std::move(next);
        m_backend->upload(Coeffs2D(m_material, m_dt));
    }

    // Return the backend.
    const Backend &Sim2D::backend() const noexcept
    {
        return *m_backend;
    }

    // Check that the source may drive the cell and is outside the PML, then keep it.
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
        if ((component == Component::Ez && edge && m_options.boundary != Boundary::Pmc) ||
            (component == Component::Hx && j + 1 == ny) ||
            (component == Component::Hy && i + 1 == nx)) {
            throw std::invalid_argument("Sim2D: source on a cell that is never updated");
        }
        const std::size_t pml = pml_thickness();
        if (i < pml || j < pml || i + pml >= nx || j + pml >= ny) {
            throw std::invalid_argument("Sim2D: source inside the PML");
        }
        m_sources.push_back({{component, m_grid.index(i, j)}, std::move(waveform)});
    }

    // Return the number of sources.
    std::size_t Sim2D::source_count() const noexcept
    {
        return m_sources.size();
    }

    // Check the cell and the step count, keep the probe and hand all probes to the backend.
    std::size_t Sim2D::add_probe(std::size_t i, std::size_t j, Component component)
    {
        if (i >= m_grid.nx() || j >= m_grid.ny()) {
            throw std::invalid_argument("Sim2D: probe outside the grid");
        }
        if (m_step_count != 0) {
            throw std::logic_error("Sim2D: probes must be added before the first run");
        }
        m_probes.push_back({component, m_grid.index(i, j)});
        m_backend->set_probes(m_probes);
        return m_probes.size() - 1;
    }

    // Return the number of probes.
    std::size_t Sim2D::probe_count() const noexcept
    {
        return m_probes.size();
    }

    // Fetch the records and pick out one probe's column.
    std::vector<FloatType> Sim2D::probe_data(std::size_t id) const
    {
        if (id >= m_probes.size()) {
            throw std::out_of_range("Sim2D: unknown probe id");
        }
        const std::vector<FloatType> records = m_backend->fetch_probes();
        const std::size_t nprobes = m_probes.size();
        std::vector<FloatType> data(records.size() / nprobes);
        for (std::size_t n = 0; n < data.size(); ++n) {
            data[n] = records[n * nprobes + id];
        }
        return data;
    }

    // Build the Yee time of every recorded step for one probe.
    std::vector<double> Sim2D::probe_times(std::size_t id) const
    {
        if (id >= m_probes.size()) {
            throw std::out_of_range("Sim2D: unknown probe id");
        }
        const double offset = m_probes[id].component == Component::Ez ? 1.0 : 0.5;
        std::vector<double> times(m_backend->probe_steps());
        for (std::size_t n = 0; n < times.size(); ++n) {
            times[n] = (static_cast<double>(n) + offset) * m_dt;
        }
        return times;
    }

    // Take the snapshot now, or keep its step in order unless it is already known.
    void Sim2D::add_snapshot(std::size_t step)
    {
        if (step < m_step_count) {
            throw std::invalid_argument("Sim2D: snapshot step already passed");
        }
        const bool taken =
            std::any_of(m_snapshots.begin(), m_snapshots.end(), [step](const Snapshot &snap) {
                return snap.step == step;
            });
        const auto it = std::lower_bound(m_snapshot_steps.begin(), m_snapshot_steps.end(), step);
        if (taken || (it != m_snapshot_steps.end() && *it == step)) {
            return;
        }
        if (step == m_step_count) {
            m_snapshots.push_back(snapshot());
            return;
        }
        m_snapshot_steps.insert(it, step);
    }

    // Download the backend's fields into a new snapshot.
    Sim2D::Snapshot Sim2D::snapshot() const
    {
        Snapshot snap{m_step_count, Fields2D(m_grid)};
        m_backend->download(snap.fields);
        return snap;
    }

    // Return the snapshots taken.
    const std::vector<Sim2D::Snapshot> &Sim2D::snapshots() const noexcept
    {
        return m_snapshots;
    }

    // Advance in pieces that end at the scheduled snapshots and take each one.
    void Sim2D::run(std::size_t nsteps)
    {
        const std::size_t end = m_step_count + nsteps;
        while (!m_snapshot_steps.empty() && m_snapshot_steps.front() <= end) {
            advance(m_snapshot_steps.front() - m_step_count);
            m_snapshots.push_back(snapshot());
            m_snapshot_steps.erase(m_snapshot_steps.begin());
        }
        advance(end - m_step_count);
    }

    // Merge sources on the same cell, sample them for the steps ahead and step the backend.
    void Sim2D::advance(std::size_t nsteps)
    {
        if (nsteps == 0) {
            return;
        }
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
