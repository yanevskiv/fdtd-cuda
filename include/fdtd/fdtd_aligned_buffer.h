/*
 * C++ header for the 64-byte aligned buffer.
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

#ifndef FDTD_ALIGNED_BUFFER_H
#define FDTD_ALIGNED_BUFFER_H

#include <algorithm>
#include <cstddef>
#include <new>
#include <utility>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** Alignment in bytes of every field and coefficient array. */
    inline constexpr std::size_t BUFFER_ALIGNMENT = 64;

    /**
     * Owning, zero-initialized, 64-byte aligned array of trivial values.
     *
     * @tparam T Element type, trivially copyable.
     */
    template <typename T> class AlignedBuffer
    {
    public:
        /** Create an empty buffer. */
        AlignedBuffer() = default;

        /**
         * Allocate a zeroed buffer.
         *
         * @param size Number of elements.
         */
        explicit inline AlignedBuffer(std::size_t size)
            : m_data(size ? static_cast<T *>(::operator new(size * sizeof(T), std::align_val_t{BUFFER_ALIGNMENT})) : nullptr)
            , m_size(size)
        {
            fill(T{});
        }

        /** Release the memory. */
        inline ~AlignedBuffer()
        {
            release();
        }

        /** Buffers are not copyable. */
        AlignedBuffer(const AlignedBuffer &) = delete;

        /** Buffers are not copyable. */
        AlignedBuffer &operator=(const AlignedBuffer &) = delete;

        /**
         * Take over another buffer's memory.
         *
         * @param other Buffer left empty.
         */
        inline AlignedBuffer(AlignedBuffer &&other) noexcept
            : m_data(std::exchange(other.m_data, nullptr))
            , m_size(std::exchange(other.m_size, 0))
        {
            // Empty.
        }

        /**
         * Release this buffer's memory and take over another's.
         *
         * @param other Buffer left empty.
         * @return This buffer.
         */
        inline AlignedBuffer &operator=(AlignedBuffer &&other) noexcept
        {
            if (this != &other) {
                release();
                m_data = std::exchange(other.m_data, nullptr);
                m_size = std::exchange(other.m_size, 0);
            }
            return *this;
        }

        /**
         * Pointer to the first element.
         *
         * @return Aligned pointer, null for an empty buffer.
         */
        inline T *data() noexcept
        {
            return m_data;
        }

        /**
         * Pointer to the first element.
         *
         * @return Aligned pointer, null for an empty buffer.
         */
        inline const T *data() const noexcept
        {
            return m_data;
        }

        /**
         * Number of elements.
         *
         * @return Element count.
         */
        inline std::size_t size() const noexcept
        {
            return m_size;
        }

        /**
         * Element at an index, unchecked.
         *
         * @param index Element index.
         * @return Reference to the element.
         */
        inline T &operator[](std::size_t index) noexcept
        {
            return m_data[index];
        }

        /**
         * Element at an index, unchecked.
         *
         * @param index Element index.
         * @return Reference to the element.
         */
        inline const T &operator[](std::size_t index) const noexcept
        {
            return m_data[index];
        }

        /**
         * Set every element to a value.
         *
         * @param value Value to store.
         */
        inline void fill(const T &value) noexcept
        {
            std::fill(m_data, m_data + m_size, value);
        }

    private:
        /** Free the memory and leave the buffer empty. */
        inline void release() noexcept
        {
            ::operator delete(m_data, std::align_val_t{BUFFER_ALIGNMENT});
            m_data = nullptr;
            m_size = 0;
        }

        /** Owned aligned memory, null when empty. */
        T *m_data = nullptr;

        /** Number of elements. */
        std::size_t m_size = 0;
    };
}

#endif
