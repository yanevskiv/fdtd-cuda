/*
 * C++ header for physical constants in SI units.
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

#ifndef FDTD_CONSTANTS_H
#define FDTD_CONSTANTS_H

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Speed of light in vacuum, in m/s; exact by the SI definition. */
    inline constexpr double C_0 = 299792458.0;

    /** Vacuum permeability, in H/m (CODATA 2022). */
    inline constexpr double MU_0 = 1.25663706127e-6;

    /** Vacuum permittivity, in F/m, derived so that C_0 = 1 / sqrt(MU_0 EPS_0) holds. */
    inline constexpr double EPS_0 = 1.0 / (MU_0 * C_0 * C_0);

    /** Impedance of free space, in ohms. */
    inline constexpr double ETA_0 = MU_0 * C_0;
}

#endif
