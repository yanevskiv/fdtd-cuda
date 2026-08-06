/*
 * C++ header for physics test support.
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

#ifndef FDTD_TEST_SUPPORT_H
#define FDTD_TEST_SUPPORT_H

#include "fdtd/TMz/fdtd_tmz_fields2d.h"
#include "fdtd/TMz/fdtd_tmz_material2d.h"
#include "fdtd/TMz/fdtd_tmz_sim2d.h"
#include "fdtd/fdtd_float.h"
#include "fdtd/fdtd_waveform.h"

#include <cstddef>
#include <span>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for test support, built on the public API only. */
    namespace test
    {
        /** A cell of the grid by column and row. */
        struct Cell
        {
            /** Column. */
            std::size_t i;

            /** Row. */
            std::size_t j;
        };

        /** Largest magnitude in a sequence and where it is. */
        struct Peak
        {
            /** Position of the largest magnitude. */
            std::size_t index;

            /** The largest magnitude. */
            double value;
        };

        /**
         * Electromagnetic energy per unit length in z, in J/m.
         *
         * Sums eps |Ez|^2 / 2 and mu (|Hx|^2 + |Hy|^2) / 2 over every cell, times dx dy.
         *
         * @param fields Fields on the material's grid.
         * @param material Per-cell permittivity and permeability.
         * @return The energy.
         * @throws std::invalid_argument When the grids differ in size.
         */
        double field_energy(const TMz::Fields2D &fields, const TMz::Material2D &material);

        /**
         * Energy per unit length in z that the Yee scheme conserves, in J/m.
         *
         * Ez at step n sits between H at n - 1/2 and n + 1/2, so the magnetic term takes
         * the product of the two: eps |Ez^n|^2 / 2 + mu H^(n-1/2) . H^(n+1/2) / 2, summed
         * over every cell, times dx dy. In a lossless closed box it stays constant to
         * rounding once the sources are off, where field_energy() of one set of fields
         * oscillates by a few percent.
         *
         * @param fields Fields after n steps; Ez and H are used.
         * @param next Fields after n + 1 steps; only H is used.
         * @param material Per-cell permittivity and permeability.
         * @return The energy.
         * @throws std::invalid_argument When the grids differ in size.
         */
        double field_energy(const TMz::Fields2D &fields, const TMz::Fields2D &next,
                            const TMz::Material2D &material);

        /**
         * Find the largest magnitude, the first one on a tie.
         *
         * @param values Values to search.
         * @return The peak.
         * @throws std::invalid_argument When the sequence is empty.
         */
        Peak find_peak(std::span<const FloatType> values);

        /**
         * Run a soft Ez point source and record Ez at some cells.
         *
         * The simulation must not have run yet; its probes and sources are added to.
         *
         * @param sim Simulation to run.
         * @param source Cell of the source.
         * @param waveform Value of the source at a time in seconds.
         * @param probes Cells to record.
         * @param nsteps Steps to run.
         * @return One trace per probe cell, one value per step.
         */
        std::vector<std::vector<FloatType>> point_source_traces(TMz::Sim2D &sim, Cell source,
                                                                Waveform waveform,
                                                                std::span<const Cell> probes,
                                                                std::size_t nsteps);
    }
}

#endif
