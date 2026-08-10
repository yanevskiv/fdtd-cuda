/*
 * C++ file for the 2D TMz field arrays.
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
#include "fdtd/TMz/fdtd_tmz_fields2d.h"

// Namespace for the 2D TMz module.
namespace fdtd::TMz
{
    // Allocate the five arrays for the grid.
    Fields2D::Fields2D(const Grid2D &grid)
        : m_grid(grid)
        , m_ez(grid.size())
        , m_hx(grid.size())
        , m_hy(grid.size())
        , m_ezx(grid.size())
        , m_ezy(grid.size())
    {
        // Empty.
    }

    // Return the grid.
    const Grid2D &Fields2D::grid() const noexcept
    {
        return m_grid;
    }

    // Return a view of Ez.
    std::span<FloatType> Fields2D::ez() noexcept
    {
        return {m_ez.data(), m_ez.size()};
    }

    // Return a read-only view of Ez.
    std::span<const FloatType> Fields2D::ez() const noexcept
    {
        return {m_ez.data(), m_ez.size()};
    }

    // Return a view of Hx.
    std::span<FloatType> Fields2D::hx() noexcept
    {
        return {m_hx.data(), m_hx.size()};
    }

    // Return a read-only view of Hx.
    std::span<const FloatType> Fields2D::hx() const noexcept
    {
        return {m_hx.data(), m_hx.size()};
    }

    // Return a view of Hy.
    std::span<FloatType> Fields2D::hy() noexcept
    {
        return {m_hy.data(), m_hy.size()};
    }

    // Return a read-only view of Hy.
    std::span<const FloatType> Fields2D::hy() const noexcept
    {
        return {m_hy.data(), m_hy.size()};
    }

    // Return a view of Ezx.
    std::span<FloatType> Fields2D::ezx() noexcept
    {
        return {m_ezx.data(), m_ezx.size()};
    }

    // Return a read-only view of Ezx.
    std::span<const FloatType> Fields2D::ezx() const noexcept
    {
        return {m_ezx.data(), m_ezx.size()};
    }

    // Return a view of Ezy.
    std::span<FloatType> Fields2D::ezy() noexcept
    {
        return {m_ezy.data(), m_ezy.size()};
    }

    // Return a read-only view of Ezy.
    std::span<const FloatType> Fields2D::ezy() const noexcept
    {
        return {m_ezy.data(), m_ezy.size()};
    }

    // Zero the five arrays.
    void Fields2D::zero() noexcept
    {
        m_ez.fill(0);
        m_hx.fill(0);
        m_hy.fill(0);
        m_ezx.fill(0);
        m_ezy.fill(0);
    }
}
