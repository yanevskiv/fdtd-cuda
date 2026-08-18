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

#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_component.h"
#include "fdtd/fdtd_float.h"

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
                };

                /** The eight coefficient arrays an update reads, laid out as the fields. */
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

                /** A cell of the grid by column and row. */
                struct CellIndex
                {
                    /** Column. */
                    int i;

                    /** Row. */
                    int j;
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
                 * Number of PML cells off the edge, in the four slabs that update_e_split_cell()
                 * takes.
                 *
                 * @param fields Fields and grid, with 2 pml below nx and ny when pml is set.
                 * @return The count, zero for a PML under two cells thick.
                 */
                FDTD_HOST_DEVICE inline int slab_cell_count(const DeviceFields &fields) noexcept
                {
                    if (fields.pml < 2) {
                        return 0;
                    }
                    const int depth = fields.pml - 1;
                    return 2 * depth * (fields.nx - 2) + 2 * depth * (fields.ny - 2 * fields.pml);
                }

                /**
                 * PML cell number t, counting the bottom slab, the top slab, then the left and
                 * right slabs between them, each row by row along x, as the CPU's
                 * update_pml_block().
                 *
                 * @param fields Fields and grid.
                 * @param t Number of the cell, below slab_cell_count().
                 * @return The cell.
                 */
                FDTD_HOST_DEVICE inline CellIndex slab_cell(const DeviceFields &fields,
                                                            int t) noexcept
                {
                    const int depth = fields.pml - 1;
                    const int width = fields.nx - 2;
                    const int band = depth * width;
                    if (t < band) {
                        return {1 + t % width, 1 + t / width};
                    }
                    if (t < 2 * band) {
                        return {1 + (t - band) % width,
                                fields.ny - fields.pml + (t - band) / width};
                    }
                    const int side = depth * (fields.ny - 2 * fields.pml);
                    const int u = t - 2 * band;
                    if (u < side) {
                        return {1 + u % depth, fields.pml + u / depth};
                    }
                    return {fields.nx - fields.pml + (u - side) % depth,
                            fields.pml + (u - side) / depth};
                }

                /**
                 * Number of cells on the edge of the grid.
                 *
                 * @param fields Fields and grid.
                 * @return 2 (nx + ny) - 4 on a grid at least 2 by 2, every cell on a thinner one.
                 */
                FDTD_HOST_DEVICE inline int edge_cell_count(const DeviceFields &fields) noexcept
                {
                    const int rows = fields.ny > 1 ? 2 : 1;
                    const int columns = fields.nx > 1 ? 2 : 1;
                    const int side = fields.ny > 2 ? fields.ny - 2 : 0;
                    return rows * fields.nx + columns * side;
                }

                /**
                 * Edge cell number t: the bottom row, the top row, then the left and right columns
                 * between them.
                 *
                 * @param fields Fields and grid.
                 * @param t Number of the cell, below edge_cell_count().
                 * @return The cell.
                 */
                FDTD_HOST_DEVICE inline CellIndex edge_cell(const DeviceFields &fields,
                                                            int t) noexcept
                {
                    if (t < fields.nx) {
                        return {t, 0};
                    }
                    const int rows = fields.ny > 1 ? 2 : 1;
                    if (t < rows * fields.nx) {
                        return {t - fields.nx, fields.ny - 1};
                    }
                    const int u = t - rows * fields.nx;
                    const int side = fields.ny - 2;
                    return {u < side ? 0 : fields.nx - 1, 1 + u % side};
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
                 * Advance Hx and Hy of one cell by one step, as the CPU's update_h_block().
                 *
                 * Hx is updated below the last row and Hy left of the last column; the others lie
                 * outside the domain and are left alone.
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficient arrays.
                 * @param i Column of the cell, inside the grid.
                 * @param j Row of the cell, inside the grid.
                 */
                FDTD_HOST_DEVICE inline void update_h_cell(const DeviceFields &fields,
                                                           const DeviceCoeffs &coeffs, int i,
                                                           int j) noexcept
                {
                    const int k = j * fields.pitch + i;
                    const FloatType *Ez = fields.ez;
                    if (j < fields.ny - 1) {
                        FloatType *Hx = fields.hx;
                        Hx[k] = read_only(coeffs.coeff_hx_1, k) * Hx[k] -
                                read_only(coeffs.coeff_hx_2, k) * fields.inv_dy *
                                    (read_only(Ez, k + fields.pitch) - read_only(Ez, k));
                    }
                    if (i < fields.nx - 1) {
                        FloatType *Hy = fields.hy;
                        Hy[k] = read_only(coeffs.coeff_hy_1, k) * Hy[k] +
                                read_only(coeffs.coeff_hy_2, k) * fields.inv_dx *
                                    (read_only(Ez, k + 1) - read_only(Ez, k));
                    }
                }

                /**
                 * Advance Ez of an interior cell by one step from the curl of H, as the CPU's
                 * update_e_interior_block().
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficient arrays.
                 * @param i Column of the cell, off the edge.
                 * @param j Row of the cell, off the edge.
                 */
                FDTD_HOST_DEVICE inline void update_e_interior_cell(const DeviceFields &fields,
                                                                    const DeviceCoeffs &coeffs,
                                                                    int i, int j) noexcept
                {
                    const int k = j * fields.pitch + i;
                    const FloatType *Hx = fields.hx;
                    const FloatType *Hy = fields.hy;
                    FloatType *Ez = fields.ez;
                    const FloatType curl =
                        (read_only(Hy, k) - read_only(Hy, k - 1)) * fields.inv_dx -
                        (read_only(Hx, k) - read_only(Hx, k - fields.pitch)) * fields.inv_dy;
                    Ez[k] = read_only(coeffs.coeff_ez_x1, k) * Ez[k] +
                            read_only(coeffs.coeff_ez_x2, k) * curl;
                }

                /**
                 * Advance Ezx and Ezy of a PML cell by one step, and Ez as their sum, as the CPU's
                 * update_split_rect().
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficient arrays.
                 * @param i Column of the cell, off the edge.
                 * @param j Row of the cell, off the edge.
                 */
                FDTD_HOST_DEVICE inline void update_e_split_cell(const DeviceFields &fields,
                                                                 const DeviceCoeffs &coeffs, int i,
                                                                 int j) noexcept
                {
                    const int k = j * fields.pitch + i;
                    const FloatType *Hx = fields.hx;
                    const FloatType *Hy = fields.hy;
                    FloatType *Ezx = fields.ezx;
                    FloatType *Ezy = fields.ezy;
                    Ezx[k] = read_only(coeffs.coeff_ez_x1, k) * Ezx[k] +
                             read_only(coeffs.coeff_ez_x2, k) * fields.inv_dx *
                                 (read_only(Hy, k) - read_only(Hy, k - 1));
                    Ezy[k] = read_only(coeffs.coeff_ez_y1, k) * Ezy[k] -
                             read_only(coeffs.coeff_ez_y2, k) * fields.inv_dy *
                                 (read_only(Hx, k) - read_only(Hx, k - fields.pitch));
                    fields.ez[k] = Ezx[k] + Ezy[k];
                }

                /**
                 * Update Ez of an edge cell, as the CPU's update_boundary_block().
                 *
                 * Under PEC and PML the cell's Ez, Ezx and Ezy are set to zero. Under PMC Ez takes
                 * the interior update with the x coefficients and every H outside the domain read
                 * as zero.
                 *
                 * @param fields Fields and grid.
                 * @param coeffs Coefficient arrays.
                 * @param i Column of the cell, on the edge.
                 * @param j Row of the cell, on the edge.
                 * @param boundary Outer boundary condition.
                 */
                FDTD_HOST_DEVICE inline void update_e_edge_cell(const DeviceFields &fields,
                                                                const DeviceCoeffs &coeffs, int i,
                                                                int j, Boundary boundary) noexcept
                {
                    const int k = j * fields.pitch + i;
                    FloatType *Ez = fields.ez;
                    if (boundary != Boundary::Pmc) {
                        Ez[k] = 0;
                        fields.ezx[k] = 0;
                        fields.ezy[k] = 0;
                        return;
                    }
                    const FloatType *Hx = fields.hx;
                    const FloatType *Hy = fields.hy;
                    const FloatType hyRight = i + 1 < fields.nx ? read_only(Hy, k) : FloatType(0);
                    const FloatType hyLeft = i > 0 ? read_only(Hy, k - 1) : FloatType(0);
                    const FloatType hxTop = j + 1 < fields.ny ? read_only(Hx, k) : FloatType(0);
                    const FloatType hxBottom =
                        j > 0 ? read_only(Hx, k - fields.pitch) : FloatType(0);
                    const FloatType curl =
                        (hyRight - hyLeft) * fields.inv_dx - (hxTop - hxBottom) * fields.inv_dy;
                    Ez[k] = read_only(coeffs.coeff_ez_x1, k) * Ez[k] +
                            read_only(coeffs.coeff_ez_x2, k) * curl;
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
