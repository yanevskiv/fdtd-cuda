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
#include "fdtd/TMz/fdtd_tmz_coeffs.h"

// Project constants for the vacuum permittivity and permeability.
#include "fdtd/fdtd_constants.h"

// Standard library for math and errors.
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Helpers private to this file.
    namespace
    {
        // Store a cell's decay and curl factors for a conductivity and eps or mu.
        void set_pair(AlignedBuffer<FloatType> &decay, AlignedBuffer<FloatType> &curl,
                      std::size_t k, double sigma, double perm, double dt) noexcept
        {
            const double a = sigma * dt / (2.0 * perm);
            decay.data()[k] = FloatType((1.0 - a) / (1.0 + a));
            curl.data()[k] = FloatType((dt / perm) / (1.0 + a));
        }
    }

    // Check dt, then fill the arrays cell by cell, adding the PML profiles to the conductivities.
    Coeffs::Coeffs(const Material &material, double dt)
        : m_grid(material.grid())
        , m_dt(dt)
        , m_pml_thickness(material.pml_thickness())
        , m_coeff_ez_x1(m_grid.size())
        , m_coeff_ez_x2(m_grid.size())
        , m_coeff_ez_y1(m_grid.size())
        , m_coeff_ez_y2(m_grid.size())
        , m_coeff_hx_1(m_grid.size())
        , m_coeff_hx_2(m_grid.size())
        , m_coeff_hy_1(m_grid.size())
        , m_coeff_hy_2(m_grid.size())
    {
        if (!std::isfinite(dt) || !(dt > 0)) {
            throw std::invalid_argument("Coeffs: time step must be positive and finite");
        }

        const auto epsR = material.eps_r();
        const auto muR = material.mu_r();
        const auto sigmaE = material.sigma_e();
        const auto sigmaM = material.sigma_m();
        const auto pmlEX = material.sigma_e_x();
        const auto pmlEY = material.sigma_e_y();
        const auto pmlMX = material.sigma_m_x();
        const auto pmlMY = material.sigma_m_y();
        for (std::size_t k = 0; k < m_grid.size(); ++k) {
            const double eps = double(epsR[k]) * EPS_0;
            const double mu = double(muR[k]) * MU_0;
            const double sigmaEx = double(sigmaE[k]) + double(epsR[k]) * double(pmlEX[k]);
            const double sigmaEy = double(sigmaE[k]) + double(epsR[k]) * double(pmlEY[k]);
            const double sigmaMx = double(sigmaM[k]) + double(muR[k]) * double(pmlMX[k]);
            const double sigmaMy = double(sigmaM[k]) + double(muR[k]) * double(pmlMY[k]);

            set_pair(m_coeff_ez_x1, m_coeff_ez_x2, k, sigmaEx, eps, dt);
            set_pair(m_coeff_ez_y1, m_coeff_ez_y2, k, sigmaEy, eps, dt);
            set_pair(m_coeff_hx_1, m_coeff_hx_2, k, sigmaMy, mu, dt);
            set_pair(m_coeff_hy_1, m_coeff_hy_2, k, sigmaMx, mu, dt);
        }
    }

    // Return the grid.
    const Grid &Coeffs::grid() const noexcept
    {
        return m_grid;
    }

    // Return the time step.
    double Coeffs::dt() const noexcept
    {
        return m_dt;
    }

    // Return the PML thickness.
    std::size_t Coeffs::pml_thickness() const noexcept
    {
        return m_pml_thickness;
    }

    // Return a read-only view of the Ez decay factor along x.
    std::span<const FloatType> Coeffs::coeff_ez_x1() const noexcept
    {
        return {m_coeff_ez_x1.data(), m_coeff_ez_x1.size()};
    }

    // Return a read-only view of the Ez curl factor along x.
    std::span<const FloatType> Coeffs::coeff_ez_x2() const noexcept
    {
        return {m_coeff_ez_x2.data(), m_coeff_ez_x2.size()};
    }

    // Return a read-only view of the Ez decay factor along y.
    std::span<const FloatType> Coeffs::coeff_ez_y1() const noexcept
    {
        return {m_coeff_ez_y1.data(), m_coeff_ez_y1.size()};
    }

    // Return a read-only view of the Ez curl factor along y.
    std::span<const FloatType> Coeffs::coeff_ez_y2() const noexcept
    {
        return {m_coeff_ez_y2.data(), m_coeff_ez_y2.size()};
    }

    // Return a read-only view of the Hx decay factor.
    std::span<const FloatType> Coeffs::coeff_hx_1() const noexcept
    {
        return {m_coeff_hx_1.data(), m_coeff_hx_1.size()};
    }

    // Return a read-only view of the Hx curl factor.
    std::span<const FloatType> Coeffs::coeff_hx_2() const noexcept
    {
        return {m_coeff_hx_2.data(), m_coeff_hx_2.size()};
    }

    // Return a read-only view of the Hy decay factor.
    std::span<const FloatType> Coeffs::coeff_hy_1() const noexcept
    {
        return {m_coeff_hy_1.data(), m_coeff_hy_1.size()};
    }

    // Return a read-only view of the Hy curl factor.
    std::span<const FloatType> Coeffs::coeff_hy_2() const noexcept
    {
        return {m_coeff_hy_2.data(), m_coeff_hy_2.size()};
    }
}
