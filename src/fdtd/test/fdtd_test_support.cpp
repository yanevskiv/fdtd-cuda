/*
 * C++ file for physics test support.
 *
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

// Function declarations.
#include "fdtd/test/fdtd_test_support.h"

// Physical constants.
#include "fdtd/fdtd_constants.h"

// Standard library for math, errors and moves.
#include <cmath>
#include <stdexcept>
#include <utility>

// Namespace for test support.
namespace fdtd::test
{
    // Helpers private to this file.
    namespace
    {
        // Check that two grids have the same size.
        void require_same_size(const TMz::Grid2D &a, const TMz::Grid2D &b)
        {
            if (a.nx() != b.nx() || a.ny() != b.ny()) {
                throw std::invalid_argument("field_energy: grids differ in size");
            }
        }

        // Sum the energy with Ez from one set of fields and the product of H from two.
        double energy(std::span<const FloatType> ez, const TMz::Fields2D &h0,
                      const TMz::Fields2D &h1, const TMz::Material2D &material)
        {
            const TMz::Grid2D &grid = material.grid();
            const auto epsR = material.eps_r();
            const auto muR = material.mu_r();
            double sum = 0;
            for (std::size_t j = 0; j < grid.ny(); ++j) {
                for (std::size_t i = 0; i < grid.nx(); ++i) {
                    const std::size_t k = grid.index(i, j);
                    const double e = double(ez[k]);
                    const double hh = double(h0.hx()[k]) * double(h1.hx()[k]) +
                                      double(h0.hy()[k]) * double(h1.hy()[k]);
                    sum += EPS_0 * double(epsR[k]) * e * e + MU_0 * double(muR[k]) * hh;
                }
            }
            return 0.5 * sum * double(grid.dx()) * double(grid.dy());
        }
    }

    // Sum the energy of one set of fields.
    double field_energy(const TMz::Fields2D &fields, const TMz::Material2D &material)
    {
        require_same_size(fields.grid(), material.grid());
        return energy(fields.ez(), fields, fields, material);
    }

    // Sum the energy with Ez from the first fields and H from the half steps around it.
    double field_energy(const TMz::Fields2D &fields, const TMz::Fields2D &next,
                        const TMz::Material2D &material)
    {
        require_same_size(fields.grid(), material.grid());
        require_same_size(next.grid(), material.grid());
        return energy(fields.ez(), fields, next, material);
    }

    // Scan for the largest magnitude.
    Peak find_peak(std::span<const FloatType> values)
    {
        if (values.empty()) {
            throw std::invalid_argument("find_peak: no values");
        }
        Peak peak{0, std::abs(double(values[0]))};
        for (std::size_t k = 1; k < values.size(); ++k) {
            const double magnitude = std::abs(double(values[k]));
            if (magnitude > peak.value) {
                peak = {k, magnitude};
            }
        }
        return peak;
    }

    // Add the source and the probes, run, and collect every probe's trace.
    std::vector<std::vector<FloatType>> point_source_traces(TMz::Sim2D &sim, Cell source,
                                                            TMz::Waveform waveform,
                                                            std::span<const Cell> probes,
                                                            std::size_t nsteps)
    {
        sim.add_source(source.i, source.j, TMz::Component::Ez, std::move(waveform));
        std::vector<std::size_t> ids;
        ids.reserve(probes.size());
        for (const Cell &cell : probes) {
            ids.push_back(sim.add_probe(cell.i, cell.j, TMz::Component::Ez));
        }
        sim.run(nsteps);
        std::vector<std::vector<FloatType>> traces;
        traces.reserve(ids.size());
        for (const std::size_t id : ids) {
            traces.push_back(sim.probe_data(id));
        }
        return traces;
    }
}
