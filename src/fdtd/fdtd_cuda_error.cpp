/*
 * C++ file for the error a failed CUDA call throws.
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

// Class declaration.
#include "fdtd/fdtd_cuda_error.h"

// Namespace for the fdtd library.
namespace fdtd
{
    // Keep the code beside the message.
    CudaError::CudaError(int code, const std::string &message)
        : std::runtime_error(message)
        , m_code(code)
    {
        // Empty.
    }

    // Return the code.
    int CudaError::code() const noexcept
    {
        return m_code;
    }
}
