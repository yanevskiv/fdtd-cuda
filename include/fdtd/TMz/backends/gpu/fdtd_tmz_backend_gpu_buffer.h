/*
 * C++ header for buffers in CUDA device memory and the copies to and from them.
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

#ifndef FDTD_TMZ_BACKEND_GPU_BUFFER_H
#define FDTD_TMZ_BACKEND_GPU_BUFFER_H

#include "fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_stream.h"

#include <cstddef>
#include <span>
#include <utility>

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
                 * Allocate device memory for an array.
                 *
                 * @param count Number of elements.
                 * @param size Size of one element in bytes.
                 * @return The memory, null for no elements.
                 * @throws std::length_error When count * size overflows.
                 * @throws CudaError When the device cannot provide the memory.
                 */
                void *allocate(std::size_t count, std::size_t size);

                /**
                 * Free device memory from allocate(), ignoring any error.
                 *
                 * @param pointer Memory to free, or null.
                 */
                void release(void *pointer) noexcept;

                /**
                 * Row pitch in elements that pads a row to a multiple of 128 bytes, the size of
                 * a fully coalesced warp access of 32 floats.
                 *
                 * @param width Elements in a row.
                 * @param size Size of one element in bytes, dividing 128.
                 * @return The pitch, at least width.
                 */
                std::size_t device_pitch(std::size_t width, std::size_t size) noexcept;

                /**
                 * Queue a copy of rows from the host to the device.
                 *
                 * Pitches and width are in bytes. The host memory may be reused when the call
                 * returns.
                 *
                 * @param target Device memory of height rows.
                 * @param targetPitch Bytes from one target row to the next.
                 * @param source Host memory of height rows.
                 * @param sourcePitch Bytes from one source row to the next.
                 * @param width Bytes to copy per row.
                 * @param height Rows to copy.
                 * @param stream Stream to queue the copy on.
                 * @throws CudaError When CUDA rejects the copy.
                 */
                void copy_rows_to_device(void *target, std::size_t targetPitch, const void *source,
                                         std::size_t sourcePitch, std::size_t width,
                                         std::size_t height, const Stream &stream);

                /**
                 * Queue a copy of rows from the device to the host, without waiting for it.
                 *
                 * Pitches and width are in bytes. The host memory holds the rows once the work
                 * queued before the copy and the copy are done; it should be page-locked (see
                 * HostPin), or the call waits as for copy_rows_to_host().
                 *
                 * @param target Host memory of height rows.
                 * @param targetPitch Bytes from one target row to the next.
                 * @param source Device memory of height rows.
                 * @param sourcePitch Bytes from one source row to the next.
                 * @param width Bytes to copy per row.
                 * @param height Rows to copy.
                 * @param stream Stream to queue the copy on.
                 * @throws CudaError When CUDA rejects the copy.
                 */
                void queue_rows_to_host(void *target, std::size_t targetPitch, const void *source,
                                        std::size_t sourcePitch, std::size_t width,
                                        std::size_t height, const Stream &stream);

                /**
                 * Queue a copy of bytes from device memory to device memory.
                 *
                 * @param target Device memory of at least bytes.
                 * @param source Device memory of at least bytes.
                 * @param bytes Bytes to copy.
                 * @param stream Stream to queue the copy on.
                 * @throws CudaError When CUDA rejects the copy.
                 */
                void copy_on_device(void *target, const void *source, std::size_t bytes,
                                    const Stream &stream);

                /**
                 * Copy rows from the device to the host, after the work queued before it.
                 *
                 * Pitches and width are in bytes. The host memory holds the rows when the call
                 * returns.
                 *
                 * @param target Host memory of height rows.
                 * @param targetPitch Bytes from one target row to the next.
                 * @param source Device memory of height rows.
                 * @param sourcePitch Bytes from one source row to the next.
                 * @param width Bytes to copy per row.
                 * @param height Rows to copy.
                 * @param stream Stream to queue the copy on.
                 * @throws CudaError When CUDA rejects the copy or earlier work failed.
                 */
                void copy_rows_to_host(void *target, std::size_t targetPitch, const void *source,
                                       std::size_t sourcePitch, std::size_t width,
                                       std::size_t height, const Stream &stream);

                /**
                 * Queue the zeroing of device memory.
                 *
                 * @param target Device memory.
                 * @param bytes Bytes to set to zero.
                 * @param stream Stream to queue the work on.
                 * @throws CudaError When CUDA rejects the work.
                 */
                void fill_zero(void *target, std::size_t bytes, const Stream &stream);

                /**
                 * An array in device memory with one owner, freed when the owner goes.
                 *
                 * Its contents start undefined; zero() clears them.
                 *
                 * @tparam T Element type, trivially copyable.
                 */
                template <typename T> class DeviceBuffer
                {
                public:
                    /** Create an empty buffer. */
                    DeviceBuffer() noexcept = default;

                    /**
                     * Allocate a buffer of a number of elements.
                     *
                     * @param size Number of elements.
                     * @throws std::length_error When the size in bytes overflows.
                     * @throws CudaError When the device cannot provide the memory.
                     */
                    inline explicit DeviceBuffer(std::size_t size)
                        : m_data(static_cast<T *>(allocate(size, sizeof(T))))
                        , m_size(size)
                    {
                        // Empty.
                    }

                    /** Free the memory. */
                    inline ~DeviceBuffer()
                    {
                        release(m_data);
                    }

                    /** Copying is not allowed: the memory has one owner. */
                    DeviceBuffer(const DeviceBuffer &) = delete;

                    /**
                     * Take the memory of another buffer, which is left empty.
                     *
                     * @param other Buffer to take from.
                     */
                    inline DeviceBuffer(DeviceBuffer &&other) noexcept
                        : m_data(std::exchange(other.m_data, nullptr))
                        , m_size(std::exchange(other.m_size, 0))
                    {
                        // Empty.
                    }

                    /** Copying is not allowed: the memory has one owner. */
                    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

                    /**
                     * Free this memory and take that of another buffer, which is left empty.
                     *
                     * @param other Buffer to take from.
                     * @return This buffer.
                     */
                    inline DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
                    {
                        if (this != &other) {
                            release(m_data);
                            m_data = std::exchange(other.m_data, nullptr);
                            m_size = std::exchange(other.m_size, 0);
                        }
                        return *this;
                    }

                    /**
                     * Device address of the first element.
                     *
                     * @return The address, null when empty.
                     */
                    inline T *data() noexcept
                    {
                        return m_data;
                    }

                    /**
                     * Device address of the first element.
                     *
                     * @return The address, null when empty.
                     */
                    inline const T *data() const noexcept
                    {
                        return m_data;
                    }

                    /**
                     * Number of elements.
                     *
                     * @return The size.
                     */
                    inline std::size_t size() const noexcept
                    {
                        return m_size;
                    }

                    /**
                     * Whether the buffer holds no element.
                     *
                     * @return True when empty.
                     */
                    inline bool empty() const noexcept
                    {
                        return m_size == 0;
                    }

                    /**
                     * Queue the zeroing of every element.
                     *
                     * @param stream Stream to queue the work on.
                     */
                    inline void zero(const Stream &stream)
                    {
                        fill_zero(m_data, m_size * sizeof(T), stream);
                    }

                private:
                    /** Device memory, null when empty. */
                    T *m_data = nullptr;

                    /** Number of elements. */
                    std::size_t m_size = 0;
                };

                /**
                 * Queue a copy of host rows into a device buffer, with pitches in elements.
                 *
                 * @tparam T Element type.
                 * @param target Device buffer of at least height rows of targetPitch.
                 * @param targetPitch Elements from one target row to the next.
                 * @param source Host rows.
                 * @param sourcePitch Elements from one source row to the next.
                 * @param width Elements to copy per row.
                 * @param height Rows to copy.
                 * @param stream Stream to queue the copy on.
                 */
                template <typename T>
                inline void upload_rows(DeviceBuffer<T> &target, std::size_t targetPitch,
                                        std::span<const T> source, std::size_t sourcePitch,
                                        std::size_t width, std::size_t height, const Stream &stream)
                {
                    copy_rows_to_device(target.data(), targetPitch * sizeof(T), source.data(),
                                        sourcePitch * sizeof(T), width * sizeof(T), height, stream);
                }

                /**
                 * Copy device rows into host memory, with pitches in elements.
                 *
                 * @tparam T Element type.
                 * @param target Host rows.
                 * @param targetPitch Elements from one target row to the next.
                 * @param source Device buffer of at least height rows of sourcePitch.
                 * @param sourcePitch Elements from one source row to the next.
                 * @param width Elements to copy per row.
                 * @param height Rows to copy.
                 * @param stream Stream to queue the copy on.
                 */
                template <typename T>
                inline void download_rows(std::span<T> target, std::size_t targetPitch,
                                          const DeviceBuffer<T> &source, std::size_t sourcePitch,
                                          std::size_t width, std::size_t height,
                                          const Stream &stream)
                {
                    copy_rows_to_host(target.data(), targetPitch * sizeof(T), source.data(),
                                      sourcePitch * sizeof(T), width * sizeof(T), height, stream);
                }

                /**
                 * Queue a copy of a host array to the start of a device buffer.
                 *
                 * @tparam T Element type.
                 * @param target Device buffer of at least source.size() elements.
                 * @param source Host array.
                 * @param stream Stream to queue the copy on.
                 */
                template <typename T>
                inline void upload(DeviceBuffer<T> &target, std::span<const T> source,
                                   const Stream &stream)
                {
                    upload_rows(target, source.size(), source, source.size(), source.size(), 1,
                                stream);
                }

                /**
                 * Copy the start of a device buffer into a host array.
                 *
                 * @tparam T Element type.
                 * @param target Host array.
                 * @param source Device buffer of at least target.size() elements.
                 * @param stream Stream to queue the copy on.
                 */
                template <typename T>
                inline void download(std::span<T> target, const DeviceBuffer<T> &source,
                                     const Stream &stream)
                {
                    download_rows(target, target.size(), source, target.size(), target.size(), 1,
                                  stream);
                }

                /**
                 * Grow a device buffer to hold at least a number of elements, dropping its
                 * contents when it grows.
                 *
                 * @tparam T Element type.
                 * @param buffer Buffer to grow.
                 * @param size Elements it must hold.
                 */
                template <typename T> inline void reserve(DeviceBuffer<T> &buffer, std::size_t size)
                {
                    if (buffer.size() < size) {
                        buffer = DeviceBuffer<T>(size);
                    }
                }
            }
        }
    }
}

#endif
