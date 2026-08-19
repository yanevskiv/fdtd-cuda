/*
 * C++ header for the 2D TMz stepper, one advance of the fields part by part.
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

#ifndef FDTD_TMZ_BACKEND_CPU_STEPPER_H
#define FDTD_TMZ_BACKEND_CPU_STEPPER_H

#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_block.h"
#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_partition.h"
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/fdtd_float.h"

#include <cstddef>
#include <span>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Namespace for the 2D TMz module. */
    namespace TMz
    {
        /** Namespace for the 2D TMz backends. */
        namespace backends
        {
            /** Namespace for what the 2D TMz CPU backends share. */
            namespace cpu
            {
                /**
                 * One advance of a CPU backend's fields, as two phases per part per step.
                 *
                 * Step n of a part is h_phase(part, n) then e_phase(part, n). The H phase updates
                 * the part's H and adds the H sources of its cells; the E phase updates its Ez,
                 * adds the Ez sources of its cells and writes the records of the probes on them.
                 * Whoever runs a phase owns the part's cells for that phase, so sources and probes
                 * need no lock: a backend only has to finish the H phase of a part's neighbours
                 * before its E phase, and the E phase of the neighbours before the next H phase.
                 * The serial backend runs the one part of Partition::whole.
                 */
                class Stepper
                {
                public:
                    /**
                     * Prepare an advance and sort the sources and probes by the part of their cell.
                     *
                     * @param fields Fields to advance; they must outlive the stepper.
                     * @param coeffs Coefficient arrays on the same grid as the fields.
                     * @param pml PML thickness in cells, zero for none.
                     * @param boundary Outer boundary condition.
                     * @param partition Parts of the grid; it must outlive the stepper.
                     * @param sources Distinct cells inside the grid to add to.
                     * @param values One value per source per step, step-major.
                     * @param probes Cells inside the grid to record.
                     * @param records Space for one value per probe per step, step-major: step n of
                     *        probe k goes to records[n * probes.size() + k].
                     */
                    Stepper(Fields &fields, CoeffsView coeffs, std::size_t pml, Boundary boundary,
                            const Partition &partition, std::span<const PointInjection> sources,
                            std::span<const FloatType> values, std::span<const PointProbe> probes,
                            FloatType *records);

                    /**
                     * Run the H phase of a part in step n: its H, then the H sources on its cells.
                     *
                     * @tparam Simd Whether the inner loops carry `omp simd`.
                     * @tparam Table Whether the core cells read the uniform tuple of the
                     *         coefficients, when they have one.
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    template <bool Simd = false, bool Table = false>
                    inline void h_phase(std::size_t part, std::size_t n) const noexcept
                    {
                        for (const Block &block : m_partition.blocks(part)) {
                            update_h_block<Simd, Table>(m_fields, m_coeffs, block);
                        }
                        add_sources(part, n, true);
                    }

                    /**
                     * Run the E phase of a part in step n: its Ez, the Ez sources on its cells,
                     * then the probes on its cells.
                     *
                     * @tparam Simd Whether the inner loops carry `omp simd`.
                     * @tparam Table Whether the core cells read the uniform tuple of the
                     *         coefficients, when they have one.
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    template <bool Simd = false, bool Table = false>
                    inline void e_phase(std::size_t part, std::size_t n) const noexcept
                    {
                        for (const Block &block : m_partition.blocks(part)) {
                            update_e_block<Simd, Table>(m_fields, m_coeffs, block, m_pml,
                                                        m_boundary);
                        }
                        add_sources(part, n, false);
                        record_probes(part, n);
                    }

                private:
                    /**
                     * Add step n's values to the sources of a part, the H ones or the Ez ones.
                     *
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     * @param magnetic Whether to add to Hx and Hy, or else to Ez.
                     */
                    void add_sources(std::size_t part, std::size_t n, bool magnetic) const noexcept;

                    /**
                     * Write step n's records of the probes of a part.
                     *
                     * @param part Part below the partition's size.
                     * @param n Step of this advance, from zero.
                     */
                    void record_probes(std::size_t part, std::size_t n) const noexcept;

                    /** Fields being advanced. */
                    Fields &m_fields;

                    /** Coefficient arrays. */
                    CoeffsView m_coeffs;

                    /** PML thickness in cells, zero for none. */
                    std::size_t m_pml;

                    /** Outer boundary condition. */
                    Boundary m_boundary;

                    /** Parts of the grid. */
                    const Partition &m_partition;

                    /** Source cells. */
                    std::span<const PointInjection> m_sources;

                    /** Source values, step-major. */
                    std::span<const FloatType> m_values;

                    /** Probe cells. */
                    std::span<const PointProbe> m_probes;

                    /** Probe records of this advance, step-major. */
                    FloatType *m_records;

                    /** Sources of part p: m_source_order from m_source_offsets[p] to [p + 1]. */
                    std::vector<std::size_t> m_source_offsets;

                    /** Source indices sorted by part. */
                    std::vector<std::size_t> m_source_order;

                    /** Probes of part p: m_probe_order from m_probe_offsets[p] to [p + 1]. */
                    std::vector<std::size_t> m_probe_offsets;

                    /** Probe indices sorted by part. */
                    std::vector<std::size_t> m_probe_order;
                };
            }
        }
    }
}

#endif
