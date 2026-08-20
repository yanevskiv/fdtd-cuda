/*
 * C++ header for the 2D TMz source waveforms sampled on the host.
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

#ifndef FDTD_TMZ_WAVEFORM_H
#define FDTD_TMZ_WAVEFORM_H

#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <functional>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** A source waveform: the source value at a time in seconds. */
        using Waveform = std::function<FloatType(FloatType)>;

        /**
         * Gaussian pulse amplitude * exp(-((t - t0) / tau)^2), evaluated in double.
         */
        class Gaussian
        {
        public:
            /**
             * Create a pulse.
             *
             * @param t0 Time of the peak in seconds, finite.
             * @param tau Width in seconds, positive and finite.
             * @param amplitude Peak value, finite.
             * @throws std::invalid_argument When a parameter is out of range.
             */
            Gaussian(double t0, double tau, double amplitude = 1);

            /**
             * Create a unit pulse whose width is a number of time steps, peaking at three widths.
             *
             * Starting three widths early keeps the first sample below exp(-9), so switching
             * the source on adds no visible step.
             *
             * @param dt Time step in seconds, positive and finite.
             * @param widthSteps Width in time steps, positive and finite.
             * @return The pulse.
             * @throws std::invalid_argument When a parameter is out of range.
             */
            static Gaussian from_steps(double dt, double widthSteps = 30);

            /**
             * Time of the peak.
             *
             * @return Time in seconds.
             */
            double t0() const noexcept;

            /**
             * Width of the pulse.
             *
             * @return Width in seconds.
             */
            double tau() const noexcept;

            /**
             * Peak value.
             *
             * @return The amplitude.
             */
            double amplitude() const noexcept;

            /**
             * Value of the pulse.
             *
             * @param t Time in seconds.
             * @return The value at t.
             */
            FloatType operator()(FloatType t) const noexcept;

        private:
            /** Time of the peak in seconds. */
            double m_t0;

            /** Width in seconds. */
            double m_tau;

            /** Peak value. */
            double m_amplitude;
        };

        /**
         * Sample a waveform once per time step: value n is taken at (firstStep + n + offset) * dt.
         *
         * Each time comes from its whole step index, so sampling steps [a, a + b) in one call
         * or in two gives the same values bit for bit. Times are computed in double and
         * converted to FloatType only for the call, so they do not drift over long runs in
         * float builds.
         *
         * @param waveform Waveform to sample.
         * @param dt Time step in seconds.
         * @param firstStep Step index of the first sample.
         * @param nsteps Number of samples.
         * @param offset Offset in steps added to every index, such as 0.5 for H.
         * @return The samples.
         * @throws std::invalid_argument When the waveform is empty.
         */
        std::vector<FloatType> sample_waveform(const Waveform &waveform, double dt,
                                               std::size_t firstStep, std::size_t nsteps,
                                               double offset = 0);
    }
}

#endif
