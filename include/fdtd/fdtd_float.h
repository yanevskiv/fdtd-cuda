/*
 * C++ header for the fdtd floating-point type.
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

#ifndef FDTD_FLOAT_H
#define FDTD_FLOAT_H

#ifndef FDTD_DOUBLE
#define FDTD_DOUBLE 0
#endif

/** Namespace for the fdtd library. */
namespace fdtd
{
#if FDTD_DOUBLE

    /** Floating-point type of fields and coefficients, double when FDTD_DOUBLE is set. */
    using FloatType = double;
#else

    /** Floating-point type of fields and coefficients, float unless FDTD_DOUBLE is set. */
    using FloatType = float;
#endif
}

#endif
