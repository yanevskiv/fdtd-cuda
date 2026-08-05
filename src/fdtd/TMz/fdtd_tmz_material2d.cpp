/*
 * C++ file for the 2D TMz material map.
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
#include "fdtd/TMz/fdtd_tmz_material2d.h"

// Standard library for math, limits and errors.
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Helpers private to this file.
    namespace
    {
        // Throw unless the medium is finite, with positive eps_r and mu_r and
        // non-negative conductivities.
        void check_medium(const Material2D::Medium &m)
        {
            const bool finite = std::isfinite(m.eps_r) && std::isfinite(m.mu_r) &&
                                std::isfinite(m.sigma_e) && std::isfinite(m.sigma_m);
            if (!finite || !(m.eps_r > 0) || !(m.mu_r > 0) || !(m.sigma_e >= 0) ||
                !(m.sigma_m >= 0)) {
                throw std::invalid_argument("Material2D: medium is not physical");
            }
        }
    }

    // Allocate the arrays and fill them with vacuum.
    Material2D::Material2D(const Grid2D &grid)
        : m_grid(grid)
        , m_eps_r(grid.size(), FloatType(1))
        , m_mu_r(grid.size(), FloatType(1))
        , m_sigma_e(grid.size(), FloatType(0))
        , m_sigma_m(grid.size(), FloatType(0))
    {}

    // Return the grid.
    const Grid2D &Material2D::grid() const noexcept
    {
        return m_grid;
    }

    // Gather the parameters of one cell after a bounds check.
    Material2D::Medium Material2D::at(std::size_t i, std::size_t j) const
    {
        if (i >= m_grid.nx() || j >= m_grid.ny()) {
            throw std::out_of_range("Material2D: cell outside the grid");
        }
        const std::size_t k = m_grid.index(i, j);
        return {m_eps_r[k], m_mu_r[k], m_sigma_e[k], m_sigma_m[k]};
    }

    // Fill the whole grid through fill_rect.
    void Material2D::fill(const Medium &medium)
    {
        fill_rect(0, 0, m_grid.nx(), m_grid.ny(), medium);
    }

    // Clip the rectangle to the grid and set its cells.
    void Material2D::fill_rect(std::size_t i0, std::size_t j0, std::size_t i1, std::size_t j1,
                               const Medium &medium)
    {
        check_medium(medium);
        i1 = std::min(i1, m_grid.nx());
        j1 = std::min(j1, m_grid.ny());
        for (std::size_t j = j0; j < j1; ++j) {
            for (std::size_t i = i0; i < i1; ++i) {
                set(m_grid.index(i, j), medium);
            }
        }
    }

    // Clip the circle's bounding box to the grid, clamping to [0, n] before the
    // conversion to an index, and set the cells inside it.
    void Material2D::fill_circle(double ci, double cj, double radius, const Medium &medium)
    {
        check_medium(medium);
        if (!std::isfinite(ci) || !std::isfinite(cj) || !std::isfinite(radius) || !(radius >= 0)) {
            throw std::invalid_argument("Material2D: circle must be finite and non-negative");
        }

        const auto lo = [](double v, std::size_t n) {
            return static_cast<std::size_t>(std::clamp(std::ceil(v), 0.0, double(n)));
        };
        const auto hi = [](double v, std::size_t n) {
            return static_cast<std::size_t>(std::clamp(std::floor(v) + 1, 0.0, double(n)));
        };
        const std::size_t i0 = lo(ci - radius, m_grid.nx());
        const std::size_t i1 = hi(ci + radius, m_grid.nx());
        const std::size_t j0 = lo(cj - radius, m_grid.ny());
        const std::size_t j1 = hi(cj + radius, m_grid.ny());
        const double r2 = radius * radius;

        for (std::size_t j = j0; j < j1; ++j) {
            const double y = double(j) - cj;
            for (std::size_t i = i0; i < i1; ++i) {
                const double x = double(i) - ci;
                if (x * x + y * y <= r2) {
                    set(m_grid.index(i, j), medium);
                }
            }
        }
    }

    // Return the relative permittivity.
    std::span<const FloatType> Material2D::eps_r() const noexcept
    {
        return m_eps_r;
    }

    // Return the relative permeability.
    std::span<const FloatType> Material2D::mu_r() const noexcept
    {
        return m_mu_r;
    }

    // Return the electric conductivity.
    std::span<const FloatType> Material2D::sigma_e() const noexcept
    {
        return m_sigma_e;
    }

    // Return the magnetic conductivity.
    std::span<const FloatType> Material2D::sigma_m() const noexcept
    {
        return m_sigma_m;
    }

    // Write the four parameters of one cell.
    void Material2D::set(std::size_t k, const Medium &medium) noexcept
    {
        m_eps_r[k] = medium.eps_r;
        m_mu_r[k] = medium.mu_r;
        m_sigma_e[k] = medium.sigma_e;
        m_sigma_m[k] = medium.sigma_m;
    }
}
