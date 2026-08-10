/*
 * C++ file for the point source sample.
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
#include "fdtd/TMz/fdtd_tmz_waveform.h"
#include "fdtd/fdtd_constants.h"

// Standard library for math and printing.
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

// Helpers private to this file.
namespace
{
    // Print a padded label, a value and its unit on one line.
    void print_line(std::string_view label, double value, std::string_view unit)
    {
        // Print the label, padded so the values line up.
        std::cout << label;

        // Print the value.
        std::cout << value;

        // Print the unit after a space.
        std::cout << unit;

        // End the line.
        std::cout << '\n';
    }
}

// Send a Gaussian pulse across a PEC box and time its arrival at a probe.
int main()
{
    // Bring the library names into scope.
    using namespace fdtd;

    // Make a 200 x 200 box of 1 mm vacuum cells with PEC walls.
    TMz::Sim2D sim(200, 200);

    // Shape a Gaussian pulse 10 steps wide.
    const TMz::Gaussian pulse = TMz::Gaussian::from_steps(sim.dt(), 10);

    // Add the pulse to Ez at the centre as a soft source.
    sim.add_source(100, 100, TMz::Component::Ez, pulse);

    // Record Ez 50 cells to the right of the source.
    const std::size_t probe = sim.add_probe(150, 100, TMz::Component::Ez);

    // Run long enough for the pulse to pass the probe.
    sim.run(250);

    // Fetch the recorded values.
    const auto values = sim.probe_data(probe);

    // Fetch the time of each value.
    const auto times = sim.probe_times(probe);

    // Start the search for the peak at the first value.
    std::size_t peak = 0;

    // Keep the value of largest magnitude.
    for (std::size_t n = 1; n < values.size(); ++n) {
        // Take this value when it beats the peak so far.
        if (std::abs(values[n]) > std::abs(values[peak])) {
            // Remember where it is.
            peak = n;
        }
    }

    // Measure the distance from the source to the probe.
    const double distance = 50 * double(sim.grid().dx());

    // Print the time step.
    print_line("dt               ", sim.dt(), " s");

    // Print when the pulse peaked at the probe.
    print_line("peak at probe    ", times[peak], " s");

    // Print how high it peaked.
    print_line("Ez at the peak   ", double(values[peak]), " V/m");

    // Print the pulse peak plus travel at C_0; a 2D soft source peaks a few % early.
    print_line("t0 + d / C_0     ", pulse.t0() + distance / C_0, " s");

    // Copy the fields back from the backend.
    sim.download();

    // Save the final Ez for tools/plot_sim.py.
    TMz::write_snapshot("point_source.bin", sim.fields(), sim.step_count(), sim.dt());

    // Say where it went.
    std::cout << "wrote point_source.bin\n";

    // Report success.
    return 0;
}
