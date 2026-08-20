/*
 * C++ header for host memory page-locked for the device to copy to and from on its own.
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

#ifndef FDTD_TMZ_BACKEND_GPU_PINNED_H
#define FDTD_TMZ_BACKEND_GPU_PINNED_H

#include <cstddef>

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
                 * Host memory that another owner allocated, page-locked while this lives, so a copy to or from it runs on the device's copy
                 * engine while the host goes on.
                 *
                 * A copy to pageable memory goes through a staging buffer and holds the host until it is done; one to page-locked memory
                 * does not.
                 */
                class HostPin
                {
                public:
                    /** Pin nothing. */
                    HostPin() noexcept = default;

                    /**
                     * Page-lock a range of host memory.
                     *
                     * @param memory Start of the range, which must outlive this.
                     * @param bytes Size of the range; zero pins nothing.
                     * @throws CudaError When CUDA cannot lock it.
                     */
                    HostPin(void *memory, std::size_t bytes);

                    /** Unlock the range, ignoring any error, since a destructor cannot throw. */
                    ~HostPin();

                    /** Copying is not allowed: the range has one pin. */
                    HostPin(const HostPin &) = delete;

                    /**
                     * Take the range of another, which is left with none.
                     *
                     * @param other Pin to take from.
                     */
                    HostPin(HostPin &&other) noexcept;

                    /** Copying is not allowed: the range has one pin. */
                    HostPin &operator=(const HostPin &) = delete;

                    /**
                     * Unlock this range and take the range of another, which is left with none.
                     *
                     * @param other Pin to take from.
                     * @return This pin.
                     */
                    HostPin &operator=(HostPin &&other) noexcept;

                private:
                    /** Start of the locked range, null for none. */
                    void *m_memory = nullptr;
                };
            }
        }
    }
}

#endif
