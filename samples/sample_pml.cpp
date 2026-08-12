/*
 * C++ file for the PML sample.
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
#include "fdtd/TMz/fdtd_tmz_simulation.h"
#include "fdtd/TMz/fdtd_tmz_waveform.h"

// Standard library for math, containers and printing.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <vector>

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

    // Run a pulse in a 60 x 60 interior inside a border and return Ez 5 cells from its left.
    std::vector<fdtd::FloatType> run_trace(const fdtd::TMz::Simulation::Options &options,
                                           std::size_t border, const char *snapshot)
    {
        // Bring the library names into scope.
        using namespace fdtd;

        // Make the square grid of the interior and the border.
        TMz::Simulation sim(60 + 2 * border, 60 + 2 * border, options);

        // Add a Gaussian pulse 8 steps wide to Ez as a soft source.
        sim.add_source(border + 20, border + 30, TMz::Component::Ez,
                       TMz::Gaussian::from_steps(sim.dt(), 8));

        // Record Ez between the source and the left edge of the interior.
        const std::size_t probe = sim.add_probe(border + 5, border + 30, TMz::Component::Ez);

        // Run long enough for the pulse and its echo to pass the probe.
        sim.run(250);

        // Save the final Ez for tools/plot_sim.py when a file is named.
        if (snapshot != nullptr) {
            // Copy the fields back from the backend.
            sim.download();

            // Write them out.
            TMz::write_snapshot(snapshot, sim.fields(), sim.step_count(), sim.dt());
        }

        // Hand back the record.
        return sim.probe_data(probe);
    }
}

// Measure a 20-cell PML at normal incidence against a domain too large to echo in time.
int main()
{
    // Bring the library names into scope.
    using namespace fdtd;

    // Ask for the PML boundary with its default grading and a target reflection of 1e-6.
    TMz::Simulation::Options pml;
    pml.boundary = TMz::Boundary::Pml;
    pml.pml_thickness = 20;

    // Record the probe with the PML around the interior.
    const auto trace = run_trace(pml, 20, "pml.bin");

    // Record it again with PEC walls 100 cells away, which reflect too late to reach it.
    const auto reference = run_trace(TMz::Simulation::Options(), 100, nullptr);

    // Find the largest value of the clean record.
    double peak = 0;

    // Find the largest difference, which is all reflection.
    double error = 0;

    // Walk both records together.
    for (std::size_t n = 0; n < trace.size(); ++n) {
        // Keep the larger peak.
        peak = std::max(peak, std::abs(double(reference[n])));

        // Keep the larger difference.
        error = std::max(error, std::abs(double(trace[n]) - double(reference[n])));
    }

    // Print the incident peak.
    print_line("incident peak    ", peak, " V/m");

    // Print the largest echo the PML let back.
    print_line("largest echo     ", error, " V/m");

    // Print the echo against the peak in decibels.
    print_line("reflection       ", 20 * std::log10(error / peak), " dB");

    // Print the target, which the discrete grading falls short of.
    print_line("target R0        ", 20 * std::log10(pml.pml_reflection), " dB");

    // Say where the final field went.
    std::cout << "wrote pml.bin\n";

    // Report success.
    return 0;
}
