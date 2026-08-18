/*
 * C++ header for the base of the 2D TMz CPU backends.
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

#ifndef FDTD_TMZ_BACKEND_CPU_BASE_H
#define FDTD_TMZ_BACKEND_CPU_BASE_H

#include "fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_block.h"
#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_coeffs.h"
#include "fdtd/TMz/fdtd_tmz_fields.h"
#include "fdtd/fdtd_aligned_buffer.h"

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
                 * What the CPU backends share: fields, coefficients and probe records in host
                 * memory.
                 *
                 * A derived backend gives its name and type and advances the fields, as a rule
                 * through a Stepper over the arrays this class holds.
                 */
                class CpuBase : public Backend
                {
                public:
                    /**
                     * Grid the backend was made for.
                     *
                     * @return The grid.
                     */
                    const Grid &grid() const noexcept override;

                    /**
                     * Copy fields into the backend.
                     *
                     * @param fields Fields on the backend's grid.
                     */
                    void upload(const Fields &fields) override;

                    /**
                     * Copy the backend's fields out.
                     *
                     * @param fields Destination on the backend's grid.
                     */
                    void download(Fields &fields) const override;

                    /**
                     * Copy update coefficients into the backend.
                     *
                     * @param coeffs Coefficients on the backend's grid.
                     */
                    void upload(const Coeffs &coeffs) override;

                    /**
                     * Set the outer boundary condition.
                     *
                     * @param boundary Outer boundary condition.
                     */
                    void set_boundary(Boundary boundary) override;

                    /**
                     * Outer boundary condition.
                     *
                     * @return The boundary.
                     */
                    Boundary boundary() const noexcept override;

                    /**
                     * Number of probes set.
                     *
                     * @return The probe count.
                     */
                    std::size_t probe_count() const noexcept override;

                    /**
                     * Number of steps recorded since the probes were set.
                     *
                     * @return The recorded step count.
                     */
                    std::size_t probe_steps() const noexcept override;

                    /**
                     * Copy the probe records out, step-major.
                     *
                     * @return The records.
                     */
                    std::vector<FloatType> fetch_probes() const override;

                    /**
                     * Modelled traffic of one step, two passes over per-cell coefficient arrays.
                     *
                     * @return two_pass_traffic() of the grid, the active PML and the boundary.
                     */
                    StepTraffic step_traffic() const noexcept override;

                protected:
                    /**
                     * Create the backend with zeroed fields.
                     *
                     * @param grid Grid the backend works on.
                     */
                    explicit CpuBase(const Grid &grid);

                    /**
                     * Replace the probes and clear the records.
                     *
                     * @param points Cells inside the grid.
                     */
                    void reset_probes(std::span<const PointProbe> points) override;

                    /**
                     * Check that coefficients were uploaded, before a step.
                     *
                     * @throws std::logic_error When none were.
                     */
                    void require_coeffs() const;

                    /**
                     * The backend's fields.
                     *
                     * @return The fields.
                     */
                    Fields &fields() noexcept;

                    /**
                     * The backend's coefficient arrays.
                     *
                     * @return A view of them.
                     */
                    CoeffsView coeffs() const noexcept;

                    /**
                     * PML thickness the steps use: that of the coefficients under Boundary::Pml.
                     *
                     * @return The thickness, zero for none.
                     */
                    std::size_t active_pml() const noexcept;

                    /**
                     * Probe cells.
                     *
                     * @return The probes.
                     */
                    std::span<const PointProbe> probes() const noexcept;

                    /**
                     * Make room for the records of nsteps more steps.
                     *
                     * @param nsteps Steps about to run.
                     * @return Where step 0 of them goes, step-major as Stepper writes them.
                     */
                    FloatType *add_records(std::size_t nsteps);

                private:
                    /** The backend's copy of the fields. */
                    Fields m_fields;

                    /** Whether coefficients were uploaded. */
                    bool m_has_coeffs = false;

                    /** PML thickness the uploaded coefficients carry, used under Boundary::Pml. */
                    std::size_t m_pml_thickness = 0;

                    /** The backend's copy of the Ez decay factors along x. */
                    AlignedBuffer<FloatType> m_coeff_ez_x1;

                    /** The backend's copy of the Ez curl factors along x. */
                    AlignedBuffer<FloatType> m_coeff_ez_x2;

                    /** The backend's copy of the Ez decay factors along y. */
                    AlignedBuffer<FloatType> m_coeff_ez_y1;

                    /** The backend's copy of the Ez curl factors along y. */
                    AlignedBuffer<FloatType> m_coeff_ez_y2;

                    /** The backend's copy of the Hx decay factors. */
                    AlignedBuffer<FloatType> m_coeff_hx_1;

                    /** The backend's copy of the Hx curl factors. */
                    AlignedBuffer<FloatType> m_coeff_hx_2;

                    /** The backend's copy of the Hy decay factors. */
                    AlignedBuffer<FloatType> m_coeff_hy_1;

                    /** The backend's copy of the Hy curl factors. */
                    AlignedBuffer<FloatType> m_coeff_hy_2;

                    /** Outer boundary condition. */
                    Boundary m_boundary = Boundary::Pec;

                    /** Probe cells. */
                    std::vector<PointProbe> m_probes;

                    /** Probe records, step-major. */
                    std::vector<FloatType> m_records;
                };
            }
        }
    }
}

#endif
