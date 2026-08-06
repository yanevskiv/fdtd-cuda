/*
 * C++ file for source waveforms sampled on the host.
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

// Class and function declarations.
#include "fdtd/fdtd_waveform.h"

// Standard library for math and errors.
#include <cmath>
#include <stdexcept>

// Namespace for the fdtd library.
namespace fdtd
{
    // Check the parameters and keep them.
    Gaussian::Gaussian(double t0, double tau, double amplitude)
        : m_t0(t0)
        , m_tau(tau)
        , m_amplitude(amplitude)
    {
        if (!std::isfinite(t0) || !std::isfinite(amplitude)) {
            throw std::invalid_argument("Gaussian: t0 and amplitude must be finite");
        }
        if (!(tau > 0) || !std::isfinite(tau)) {
            throw std::invalid_argument("Gaussian: tau must be positive and finite");
        }
    }

    // Scale the width by dt and put the peak three widths in.
    Gaussian Gaussian::from_steps(double dt, double widthSteps)
    {
        if (!(dt > 0) || !std::isfinite(dt)) {
            throw std::invalid_argument("Gaussian: dt must be positive and finite");
        }
        const double tau = widthSteps * dt;
        return Gaussian(3 * tau, tau);
    }

    // Return the peak time.
    double Gaussian::t0() const noexcept
    {
        return m_t0;
    }

    // Return the width.
    double Gaussian::tau() const noexcept
    {
        return m_tau;
    }

    // Return the amplitude.
    double Gaussian::amplitude() const noexcept
    {
        return m_amplitude;
    }

    // Evaluate in double and convert once.
    FloatType Gaussian::operator()(FloatType t) const noexcept
    {
        const double x = (double(t) - m_t0) / m_tau;
        return FloatType(m_amplitude * std::exp(-x * x));
    }

    // Take one sample per step at (firstStep + n + offset) * dt.
    std::vector<FloatType> sample_waveform(const Waveform &waveform, double dt,
                                           std::size_t firstStep, std::size_t nsteps, double offset)
    {
        if (!waveform) {
            throw std::invalid_argument("sample_waveform: empty waveform");
        }
        std::vector<FloatType> samples(nsteps);
        for (std::size_t n = 0; n < nsteps; ++n) {
            samples[n] = waveform(FloatType((double(firstStep + n) + offset) * dt));
        }
        return samples;
    }
}
