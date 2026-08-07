/*
 * C++ file for the point source example.
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

// The simulation, its output files and the source waveform.
#include "fdtd/TMz/fdtd_tmz_output.h"
#include "fdtd/TMz/fdtd_tmz_sim2d.h"
#include "fdtd/fdtd_constants.h"
#include "fdtd/fdtd_waveform.h"

// Standard library for math and printing.
#include <cmath>
#include <cstddef>
#include <iostream>

// Send a Gaussian pulse from the centre of a 200 x 200 PEC box, record Ez 50 cells away,
// print when the peak arrives against the pulse peak plus the travel time, and save the
// final Ez. The peak lands a few percent early: a soft source acts as a current, whose
// field in 2D is about the half derivative of the pulse, and that peaks before it.
int main()
{
    using namespace fdtd;

    TMz::Sim2D sim(200, 200);
    const Gaussian pulse = Gaussian::from_steps(sim.dt(), 10);
    sim.add_source(100, 100, TMz::Component::Ez, pulse);
    const std::size_t probe = sim.add_probe(150, 100, TMz::Component::Ez);
    sim.run(250);

    const auto values = sim.probe_data(probe);
    const auto times = sim.probe_times(probe);
    std::size_t peak = 0;
    for (std::size_t n = 1; n < values.size(); ++n) {
        if (std::abs(values[n]) > std::abs(values[peak])) {
            peak = n;
        }
    }
    const double distance = 50 * double(sim.grid().dx());
    std::cout << "dt               " << sim.dt() << " s\n"
              << "peak at probe    " << times[peak] << " s, Ez " << values[peak] << " V/m\n"
              << "t0 + d / C_0     " << pulse.t0() + distance / C_0 << " s\n";

    sim.download();
    TMz::write_snapshot("point_source.bin", sim.fields(), sim.step_count(), sim.dt());
    std::cout << "wrote point_source.bin\n";
    return 0;
}
