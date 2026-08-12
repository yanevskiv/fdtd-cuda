/*
 * C++ header for the 2D TMz field arrays.
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

#ifndef FDTD_TMZ_FIELDS_H
#define FDTD_TMZ_FIELDS_H

#include "fdtd/TMz/fdtd_tmz_grid.h"
#include "fdtd/fdtd_aligned_buffer.h"
#include "fdtd/fdtd_float.h"

#include <span>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /**
         * The TMz fields Ez, Hx and Hy over a grid, with the split parts Ezx and Ezy of
         * Ez, as five aligned arrays.
         *
         * Each array has grid().size() elements, indexed with Grid::index, and starts
         * zeroed. The Yee staggering is in the update, not the storage. Ezx and Ezy span
         * the whole grid but are only kept inside a PML, where Ez = Ezx + Ezy; elsewhere
         * they stay zero.
         */
        class Fields
        {
        public:
            /**
             * Allocate zeroed fields for a grid.
             *
             * @param grid Grid the fields live on.
             */
            explicit Fields(const Grid &grid);

            /**
             * Grid the fields live on.
             *
             * @return The grid.
             */
            const Grid &grid() const noexcept;

            /**
             * Electric field along z.
             *
             * @return View of the Ez array.
             */
            std::span<FloatType> ez() noexcept;

            /**
             * Electric field along z.
             *
             * @return Read-only view of the Ez array.
             */
            std::span<const FloatType> ez() const noexcept;

            /**
             * Magnetic field along x.
             *
             * @return View of the Hx array.
             */
            std::span<FloatType> hx() noexcept;

            /**
             * Magnetic field along x.
             *
             * @return Read-only view of the Hx array.
             */
            std::span<const FloatType> hx() const noexcept;

            /**
             * Magnetic field along y.
             *
             * @return View of the Hy array.
             */
            std::span<FloatType> hy() noexcept;

            /**
             * Magnetic field along y.
             *
             * @return Read-only view of the Hy array.
             */
            std::span<const FloatType> hy() const noexcept;

            /**
             * Part of Ez driven by the x derivative of Hy, kept inside a PML.
             *
             * @return View of the Ezx array.
             */
            std::span<FloatType> ezx() noexcept;

            /**
             * Part of Ez driven by the x derivative of Hy, kept inside a PML.
             *
             * @return Read-only view of the Ezx array.
             */
            std::span<const FloatType> ezx() const noexcept;

            /**
             * Part of Ez driven by the y derivative of Hx, kept inside a PML.
             *
             * @return View of the Ezy array.
             */
            std::span<FloatType> ezy() noexcept;

            /**
             * Part of Ez driven by the y derivative of Hx, kept inside a PML.
             *
             * @return Read-only view of the Ezy array.
             */
            std::span<const FloatType> ezy() const noexcept;

            /** Set every field to zero, the split parts included. */
            void zero() noexcept;

        private:
            /** Grid the fields live on. */
            Grid m_grid;

            /** Electric field along z. */
            AlignedBuffer<FloatType> m_ez;

            /** Magnetic field along x. */
            AlignedBuffer<FloatType> m_hx;

            /** Magnetic field along y. */
            AlignedBuffer<FloatType> m_hy;

            /** Part of Ez driven by the x derivative of Hy. */
            AlignedBuffer<FloatType> m_ezx;

            /** Part of Ez driven by the y derivative of Hx. */
            AlignedBuffer<FloatType> m_ezy;
        };
    }
}

#endif
