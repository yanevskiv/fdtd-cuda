/*
 * C++ header for the 2D TMz outer boundary types.
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

#ifndef FDTD_TMZ_BOUNDARY_H
#define FDTD_TMZ_BOUNDARY_H

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Condition on the outer edge of the domain. */
        enum class Boundary {
            /** Perfect electric conductor: the edge Ez is held at zero. The default. */
            Pec,

            /** Perfect magnetic conductor: the edge Ez is updated with zero H outside. */
            Pmc,

            /**
             * Split-field PML in front of a PEC wall, set by the pml_* options of Sim2D.
             * With a thickness of zero it is the same as Pec.
             */
            Pml
        };
    }
}

#endif
