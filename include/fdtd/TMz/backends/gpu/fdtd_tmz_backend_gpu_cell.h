/*
 * C++ header for the 2D TMz update of one cell, for the host and the CUDA device.
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

#ifndef FDTD_TMZ_BACKEND_GPU_CELL_H
#define FDTD_TMZ_BACKEND_GPU_CELL_H

#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_table.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_component.h"
#include "fdtd/fdtd_float.h"

#include <cstdint>

#ifdef __CUDACC__

/** Mark a function for the host and the device under nvcc. */
#define FDTD_HOST_DEVICE __host__ __device__

#else

/** Mark a function for the host and the device; nothing outside nvcc. */
#define FDTD_HOST_DEVICE

#endif

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /** Namespace for what the 2D TMz CUDA backend's files share. */
            namespace gpu
            {
                /**
                 * The fields and grid an update reads and writes, as raw pointers and int indices.
                 *
                 * Cell (i, j) of each array is at j * pitch + i. The pointers are to device memory
                 * in the CUDA backend and to host memory in the tests.
                 */
                struct DeviceFields
                {
                    /** Ez array. */
                    FloatType *ez;

                    /** Hx array. */
                    FloatType *hx;

                    /** Hy array. */
                    FloatType *hy;

                    /** Ezx array, used in the PML only. */
                    FloatType *ezx;

                    /** Ezy array, used in the PML only. */
                    FloatType *ezy;

                    /** Cells along x. */
                    int nx;

                    /** Cells along y. */
                    int ny;

                    /** Elements from one row to the next, at least nx. */
                    int pitch;

                    /** 1 / dx, computed as FloatType(1) / dx, as the CPU kernels do. */
                    FloatType inv_dx;

                    /** 1 / dy, computed as FloatType(1) / dy, as the CPU kernels do. */
                    FloatType inv_dy;

                    /** Thickness of the active PML in cells, zero for none. */
                    int pml;

                    /** Ez array a one-pass kernel writes the step's Ez to, or null. */
                    FloatType *ez_next = nullptr;

                    /** Hx array a one-pass kernel writes the step's Hx to, or null. */
                    FloatType *hx_next = nullptr;

                    /** Hy array a one-pass kernel writes the step's Hy to, or null. */
                    FloatType *hy_next = nullptr;
                };

                /** Where the updates read a cell's coefficients. */
                enum class CoeffSource {
                    /** The eight arrays. */
                    Arrays,

                    /** One tuple for every core cell, as is_core_cell() finds them; the arrays
                     * for the others. */
                    Uniform,

                    /** The table entry of the cell's index, or the arrays for CoeffTable::ARRAYS.
                     */
                    Indexed
                };

                /**
                 * The coefficients an update reads: eight arrays laid out as the fields, and the
                 * table that stands in for them outside the PML when the source says so.
                 */
                struct DeviceCoeffs
                {
                    /** Ez decay factors along x. */
                    const FloatType *coeff_ez_x1;

                    /** Ez curl factors along x. */
                    const FloatType *coeff_ez_x2;

                    /** Ez decay factors along y. */
                    const FloatType *coeff_ez_y1;

                    /** Ez curl factors along y. */
                    const FloatType *coeff_ez_y2;

                    /** Hx decay factors. */
                    const FloatType *coeff_hx_1;

                    /** Hx curl factors. */
                    const FloatType *coeff_hx_2;

                    /** Hy decay factors. */
                    const FloatType *coeff_hy_1;

                    /** Hy curl factors. */
                    const FloatType *coeff_hy_2;

                    /** Where the updates read a cell's coefficients. */
                    CoeffSource source = CoeffSource::Arrays;

                    /** Tuple of every core cell, under CoeffSource::Uniform. */
                    CoeffTuple uniform = {};

                    /** Table entries, under CoeffSource::Indexed. */
                    const CoeffTuple *table = nullptr;

                    /** Entry of each cell, laid out as the fields, under CoeffSource::Indexed. */
                    const std::uint8_t *index = nullptr;
                };

                /** A cell of one component that a source adds to or a probe reads, on the device.
                 */
                struct DevicePoint
                {
                    /** Component the cell belongs to. */
                    Component component;

                    /** Index of the cell, j * pitch + i with the device pitch. */
                    int cell;

                    /** Column of the point in a step's row of source values or probe records. */
                    int column;
                };

                /**
                 * Array of a component.
                 *
                 * @param fields Fields and grid.
                 * @param component Ez, Hx or Hy.
                 * @return The array.
                 */
                FDTD_HOST_DEVICE inline FloatType *field_of(const DeviceFields &fields,
                                                            Component component) noexcept
                {
                    switch (component) {
                        case Component::Hx:
                            return fields.hx;
                        case Component::Hy:
                            return fields.hy;
                        case Component::Ez:
                            break;
                    }
                    return fields.ez;
                }

                /**
                 * Read an element that no thread of the running kernel writes, through the
                 * read-only data cache on the device and as a plain load on the host.
                 *
                 * @param array Array to read.
                 * @param k Index of the element.
                 * @return The element.
                 */
                FDTD_HOST_DEVICE inline FloatType read_only(const FloatType *array, int k) noexcept
                {
#ifdef __CUDA_ARCH__
                    return __ldg(array + k);
#else
                    return array[k];
#endif
                }

                /** The table's tuple of a cell, when it has one. */
                struct CellTuple
                {
                    /** Whether the cell reads the tuple rather than the arrays. */
                    bool found;

                    /** The tuple, when found. */
                    CoeffTuple tuple;
                };

                /**
                 * Lower bound of the interior along both axes.
                 *
                 * @param fields Fields and grid.
                 * @return max(pml, 1): the interior is lo <= i < nx - lo, and the same along j.
                 */
                FDTD_HOST_DEVICE inline int interior_lo(const DeviceFields &fields) noexcept
                {
                    return fields.pml > 1 ? fields.pml : 1;
                }

                /**
                 * Whether a cell lies on the edge of the grid.
                 *
                 * @param fields Fields and grid.
                 * @param i Column of the cell.
                 * @param j Row of the cell.
                 * @return True on the first or last row or column.
                 */
                FDTD_HOST_DEVICE inline bool is_edge_cell(const DeviceFields &fields, int i,
                                                          int j) noexcept
                {
                    return i == 0 || j == 0 || i == fields.nx - 1 || j == fields.ny - 1;
                }

                /**
                 * Whether a cell takes the plain Ez update: inside the PML and off the edge, as
                 * the CPU kernels' interior_block().
                 *
                 * @param fields Fields and grid.
                 * @param i Column of the cell.
                 * @param j Row of the cell.
                 * @return True for max(pml, 1) <= i < nx - max(pml, 1), and the same along j.
                 */
                FDTD_HOST_DEVICE inline bool is_interior_cell(const DeviceFields &fields, int i,
                                                              int j) noexcept
                {
                    const int lo = interior_lo(fields);
                    return i >= lo && i < fields.nx - lo && j >= lo && j < fields.ny - lo;
                }

                /**
                 * Whether a cell lies in the core of the interior: the interior less its last
                 * column and row, where Hy and Hx sit half a cell into the PML (see
                 * CoeffTable::uniform_entry()).
                 *
                 * @param fields Fields and grid.
                 * @param i Column of the cell.
                 * @param j Row of the cell.
                 * @return True for max(pml, 1) <= i < nx - max(pml, 1) - 1, and the same along j.
                 */
                FDTD_HOST_DEVICE inline bool is_core_cell(const DeviceFields &fields, int i,
                                                          int j) noexcept
                {
                    const int lo = interior_lo(fields);
                    return i >= lo && i < fields.nx - lo - 1 && j >= lo && j < fields.ny - lo - 1;
                }

                /**
                 * The table's tuple of a cell, under the source of the coefficients.
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficients and their source.
                 * @param k Index of the cell, j * pitch + i.
                 * @param i Column of the cell.
                 * @param j Row of the cell.
                 * @return The tuple, or not found when the cell reads the arrays.
                 */
                FDTD_HOST_DEVICE inline CellTuple cell_tuple(const DeviceFields &fields,
                                                             const DeviceCoeffs &coeffs, int k,
                                                             int i, int j) noexcept
                {
                    if (coeffs.source == CoeffSource::Uniform) {
                        if (is_core_cell(fields, i, j)) {
                            return {
                                .found = true,
                                .tuple = coeffs.uniform,
                            };
                        }
                    } else if (coeffs.source == CoeffSource::Indexed) {
#ifdef __CUDA_ARCH__
                        const std::uint8_t entry = __ldg(coeffs.index + k);
#else
                        const std::uint8_t entry = coeffs.index[k];
#endif
                        if (entry != CoeffTable::ARRAYS) {
                            return {
                                .found = true,
                                .tuple = coeffs.table[entry],
                            };
                        }
                    }
                    return {
                        .found = false,
                        .tuple = {},
                    };
                }

                /** New Hx and Hy of a cell. */
                struct HValues
                {
                    /** Hx. */
                    FloatType hx;

                    /** Hy. */
                    FloatType hy;
                };

                /** The H around a cell's Ez that its update reads. */
                struct HAround
                {
                    /** Hy of the cell, on its right. */
                    FloatType hy;

                    /** Hy of the cell on the left. */
                    FloatType hy_left;

                    /** Hx of the cell, above it. */
                    FloatType hx;

                    /** Hx of the cell below. */
                    FloatType hx_below;
                };

                /** New Ezx and Ezy of a PML cell. */
                struct EzSplit
                {
                    /** Ezx. */
                    FloatType ezx;

                    /** Ezy. */
                    FloatType ezy;
                };

                /**
                 * New Hx and Hy of one cell after one step, as the CPU's update_h_block() makes
                 * them, from the fields as they are.
                 *
                 * Hx is updated below the last row and Hy left of the last column; the others lie
                 * outside the domain and keep their value.
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficients and their source.
                 * @param i Column of the cell, inside the grid.
                 * @param j Row of the cell, inside the grid.
                 * @return The new values.
                 */
                FDTD_HOST_DEVICE inline HValues h_values(const DeviceFields &fields,
                                                         const DeviceCoeffs &coeffs, int i,
                                                         int j) noexcept
                {
                    const int k = j * fields.pitch + i;
                    const CellTuple table = cell_tuple(fields, coeffs, k, i, j);
                    const FloatType *Ez = fields.ez;
                    HValues h{
                        .hx = fields.hx[k],
                        .hy = fields.hy[k],
                    };
                    if (j < fields.ny - 1) {
                        const FloatType c1 =
                            table.found ? table.tuple.hx_1 : read_only(coeffs.coeff_hx_1, k);
                        const FloatType c2 =
                            table.found ? table.tuple.hx_2 : read_only(coeffs.coeff_hx_2, k);
                        h.hx = c1 * h.hx - c2 * fields.inv_dy *
                                               (read_only(Ez, k + fields.pitch) - read_only(Ez, k));
                    }
                    if (i < fields.nx - 1) {
                        const FloatType c1 =
                            table.found ? table.tuple.hy_1 : read_only(coeffs.coeff_hy_1, k);
                        const FloatType c2 =
                            table.found ? table.tuple.hy_2 : read_only(coeffs.coeff_hy_2, k);
                        h.hy = c1 * h.hy +
                               c2 * fields.inv_dx * (read_only(Ez, k + 1) - read_only(Ez, k));
                    }
                    return h;
                }

                /**
                 * Advance Hx and Hy of one cell by one step, as the CPU's update_h_block().
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficients and their source.
                 * @param i Column of the cell, inside the grid.
                 * @param j Row of the cell, inside the grid.
                 */
                FDTD_HOST_DEVICE inline void update_h_cell(const DeviceFields &fields,
                                                           const DeviceCoeffs &coeffs, int i,
                                                           int j) noexcept
                {
                    const int k = j * fields.pitch + i;
                    const HValues h = h_values(fields, coeffs, i, j);
                    if (j < fields.ny - 1) {
                        fields.hx[k] = h.hx;
                    }
                    if (i < fields.nx - 1) {
                        fields.hy[k] = h.hy;
                    }
                }

                /**
                 * The H around the Ez of a cell off the edge, read from the fields.
                 *
                 * @param fields Fields and grid.
                 * @param k Index of the cell, off the edge.
                 * @return The four values.
                 */
                FDTD_HOST_DEVICE inline HAround h_around(const DeviceFields &fields, int k) noexcept
                {
                    return {
                        .hy = read_only(fields.hy, k),
                        .hy_left = read_only(fields.hy, k - 1),
                        .hx = read_only(fields.hx, k),
                        .hx_below = read_only(fields.hx, k - fields.pitch),
                    };
                }

                /**
                 * The H around the Ez of an edge cell, read from the fields, with every H outside
                 * the domain read as zero.
                 *
                 * @param fields Fields and grid.
                 * @param i Column of the cell.
                 * @param j Row of the cell.
                 * @return The four values.
                 */
                FDTD_HOST_DEVICE inline HAround h_around_edge(const DeviceFields &fields, int i,
                                                              int j) noexcept
                {
                    const int k = j * fields.pitch + i;
                    return {
                        .hy = i + 1 < fields.nx ? read_only(fields.hy, k) : FloatType(0),
                        .hy_left = i > 0 ? read_only(fields.hy, k - 1) : FloatType(0),
                        .hx = j + 1 < fields.ny ? read_only(fields.hx, k) : FloatType(0),
                        .hx_below = j > 0 ? read_only(fields.hx, k - fields.pitch) : FloatType(0),
                    };
                }

                /**
                 * New Ez of a cell by the interior update, from its Ez and the H around it, as the
                 * CPU's update_e_interior_block(); the PMC edge takes it too.
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficients and their source.
                 * @param i Column of the cell.
                 * @param j Row of the cell.
                 * @param ez Ez of the cell.
                 * @param h The H around the cell.
                 * @return The new Ez.
                 */
                FDTD_HOST_DEVICE inline FloatType e_interior_value(const DeviceFields &fields,
                                                                   const DeviceCoeffs &coeffs,
                                                                   int i, int j, FloatType ez,
                                                                   const HAround &h) noexcept
                {
                    const int k = j * fields.pitch + i;
                    const FloatType curl =
                        (h.hy - h.hy_left) * fields.inv_dx - (h.hx - h.hx_below) * fields.inv_dy;
                    const CellTuple table = cell_tuple(fields, coeffs, k, i, j);
                    const FloatType c1 =
                        table.found ? table.tuple.ez_1 : read_only(coeffs.coeff_ez_x1, k);
                    const FloatType c2 =
                        table.found ? table.tuple.ez_2 : read_only(coeffs.coeff_ez_x2, k);
                    return c1 * ez + c2 * curl;
                }

                /**
                 * New Ezx and Ezy of a PML cell, from their values and the H around it, as the
                 * CPU's update_split_rect().
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficients and their source.
                 * @param i Column of the cell.
                 * @param j Row of the cell.
                 * @param ezx Ezx of the cell.
                 * @param ezy Ezy of the cell.
                 * @param h The H around the cell.
                 * @return The new Ezx and Ezy, whose sum is the new Ez.
                 */
                FDTD_HOST_DEVICE inline EzSplit e_split_values(const DeviceFields &fields,
                                                               const DeviceCoeffs &coeffs, int i,
                                                               int j, FloatType ezx, FloatType ezy,
                                                               const HAround &h) noexcept
                {
                    const int k = j * fields.pitch + i;
                    const CellTuple table = cell_tuple(fields, coeffs, k, i, j);
                    const FloatType x1 =
                        table.found ? table.tuple.ez_1 : read_only(coeffs.coeff_ez_x1, k);
                    const FloatType x2 =
                        table.found ? table.tuple.ez_2 : read_only(coeffs.coeff_ez_x2, k);
                    const FloatType y1 =
                        table.found ? table.tuple.ez_1 : read_only(coeffs.coeff_ez_y1, k);
                    const FloatType y2 =
                        table.found ? table.tuple.ez_2 : read_only(coeffs.coeff_ez_y2, k);
                    return {
                        .ezx = x1 * ezx + x2 * fields.inv_dx * (h.hy - h.hy_left),
                        .ezy = y1 * ezy - y2 * fields.inv_dy * (h.hx - h.hx_below),
                    };
                }

                /**
                 * Advance Ez of an interior cell by one step from the curl of H, as the CPU's
                 * update_e_interior_block().
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficients and their source.
                 * @param i Column of the cell, off the edge.
                 * @param j Row of the cell, off the edge.
                 */
                FDTD_HOST_DEVICE inline void update_e_interior_cell(const DeviceFields &fields,
                                                                    const DeviceCoeffs &coeffs,
                                                                    int i, int j) noexcept
                {
                    const int k = j * fields.pitch + i;
                    fields.ez[k] =
                        e_interior_value(fields, coeffs, i, j, fields.ez[k], h_around(fields, k));
                }

                /**
                 * Advance Ezx and Ezy of a PML cell by one step, and Ez as their sum, as the CPU's
                 * update_split_rect().
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficients and their source.
                 * @param i Column of the cell, off the edge.
                 * @param j Row of the cell, off the edge.
                 */
                FDTD_HOST_DEVICE inline void update_e_split_cell(const DeviceFields &fields,
                                                                 const DeviceCoeffs &coeffs, int i,
                                                                 int j) noexcept
                {
                    const int k = j * fields.pitch + i;
                    const EzSplit split = e_split_values(fields, coeffs, i, j, fields.ezx[k],
                                                         fields.ezy[k], h_around(fields, k));
                    fields.ezx[k] = split.ezx;
                    fields.ezy[k] = split.ezy;
                    fields.ez[k] = split.ezx + split.ezy;
                }

                /**
                 * Update Ez of an edge cell, as the CPU's update_boundary_block().
                 *
                 * Under PEC and PML the cell's Ez, Ezx and Ezy are set to zero. Under PMC Ez takes
                 * the interior update with the x coefficients and every H outside the domain read
                 * as zero.
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficients and their source.
                 * @param i Column of the cell, on the edge.
                 * @param j Row of the cell, on the edge.
                 * @param boundary Outer boundary condition.
                 */
                FDTD_HOST_DEVICE inline void update_e_edge_cell(const DeviceFields &fields,
                                                                const DeviceCoeffs &coeffs, int i,
                                                                int j, Boundary boundary) noexcept
                {
                    const int k = j * fields.pitch + i;
                    if (boundary != Boundary::Pmc) {
                        fields.ez[k] = 0;
                        fields.ezx[k] = 0;
                        fields.ezy[k] = 0;
                        return;
                    }
                    fields.ez[k] = e_interior_value(fields, coeffs, i, j, fields.ez[k],
                                                    h_around_edge(fields, i, j));
                }

                /**
                 * New Ez of any cell after one step, from the H around it: the edge, PML or
                 * interior update, whichever the cell takes. A PML cell's Ezx and Ezy are updated
                 * in place, and an edge cell's set to zero under PEC and PML.
                 *
                 * @param fields Fields and grid; Ez is read from fields.ez.
                 * @param coeffs Coefficients and their source.
                 * @param i Column of the cell, inside the grid.
                 * @param j Row of the cell, inside the grid.
                 * @param boundary Outer boundary condition.
                 * @param h The H around the cell, zero outside the domain.
                 * @return The new Ez.
                 */
                FDTD_HOST_DEVICE inline FloatType e_value(const DeviceFields &fields,
                                                          const DeviceCoeffs &coeffs, int i, int j,
                                                          Boundary boundary,
                                                          const HAround &h) noexcept
                {
                    const int k = j * fields.pitch + i;
                    if (is_edge_cell(fields, i, j)) {
                        if (boundary != Boundary::Pmc) {
                            fields.ezx[k] = 0;
                            fields.ezy[k] = 0;
                            return 0;
                        }
                        return e_interior_value(fields, coeffs, i, j, fields.ez[k], h);
                    }
                    if (is_interior_cell(fields, i, j)) {
                        return e_interior_value(fields, coeffs, i, j, fields.ez[k], h);
                    }
                    const EzSplit split =
                        e_split_values(fields, coeffs, i, j, fields.ezx[k], fields.ezy[k], h);
                    fields.ezx[k] = split.ezx;
                    fields.ezy[k] = split.ezy;
                    return split.ezx + split.ezy;
                }

                /**
                 * Advance Ez of any cell by one step: the edge, PML or interior update, whichever
                 * the cell takes.
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficient arrays.
                 * @param i Column of the cell, inside the grid.
                 * @param j Row of the cell, inside the grid.
                 * @param boundary Outer boundary condition.
                 */
                FDTD_HOST_DEVICE inline void update_e_cell(const DeviceFields &fields,
                                                           const DeviceCoeffs &coeffs, int i, int j,
                                                           Boundary boundary) noexcept
                {
                    if (is_edge_cell(fields, i, j)) {
                        update_e_edge_cell(fields, coeffs, i, j, boundary);
                    } else if (is_interior_cell(fields, i, j)) {
                        update_e_interior_cell(fields, coeffs, i, j);
                    } else {
                        update_e_split_cell(fields, coeffs, i, j);
                    }
                }
            }
        }
    }
}

#endif
