/*
 * C++ file for the 2D TMz update coefficients.
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
#include "fdtd/TMz/fdtd_tmz_coeffs2d.h"

// Project constants for the vacuum permittivity and permeability.
#include "fdtd/fdtd_constants.h"

// Standard library for math and errors.
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Check the time step, allocate the arrays and fill them cell by cell.
    Coeffs2D::Coeffs2D(const Material2D &material, double dt)
        : m_grid(material.grid())
        , m_dt(dt)
        , m_ce1(m_grid.size())
        , m_ce2(m_grid.size())
        , m_ch1(m_grid.size())
        , m_ch2(m_grid.size())
    {
        if (!std::isfinite(dt) || !(dt > 0)) {
            throw std::invalid_argument("Coeffs2D: time step must be positive and finite");
        }

        const auto epsR = material.eps_r();
        const auto muR = material.mu_r();
        const auto sigmaE = material.sigma_e();
        const auto sigmaM = material.sigma_m();
        for (std::size_t k = 0; k < m_grid.size(); ++k) {
            const double eps = double(epsR[k]) * EPS_0;
            const double mu = double(muR[k]) * MU_0;
            const double a = double(sigmaE[k]) * dt / (2.0 * eps);
            const double b = double(sigmaM[k]) * dt / (2.0 * mu);

            m_ce1.data()[k] = FloatType((1.0 - a) / (1.0 + a));
            m_ce2.data()[k] = FloatType((dt / eps) / (1.0 + a));
            m_ch1.data()[k] = FloatType((1.0 - b) / (1.0 + b));
            m_ch2.data()[k] = FloatType((dt / mu) / (1.0 + b));
        }
    }

    // Return the grid.
    const Grid2D &Coeffs2D::grid() const noexcept
    {
        return m_grid;
    }

    // Return the time step.
    double Coeffs2D::dt() const noexcept
    {
        return m_dt;
    }

    // Return a read-only view of Ce1.
    std::span<const FloatType> Coeffs2D::ce1() const noexcept
    {
        return {m_ce1.data(), m_ce1.size()};
    }

    // Return a read-only view of Ce2.
    std::span<const FloatType> Coeffs2D::ce2() const noexcept
    {
        return {m_ce2.data(), m_ce2.size()};
    }

    // Return a read-only view of Ch1.
    std::span<const FloatType> Coeffs2D::ch1() const noexcept
    {
        return {m_ch1.data(), m_ch1.size()};
    }

    // Return a read-only view of Ch2.
    std::span<const FloatType> Coeffs2D::ch2() const noexcept
    {
        return {m_ch2.data(), m_ch2.size()};
    }
}
