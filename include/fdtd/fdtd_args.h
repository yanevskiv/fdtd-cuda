/*
 * C++ header for command-line argument parsing shared by the programs.
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

#ifndef FDTD_ARGS_H
#define FDTD_ARGS_H

#include "fdtd/TMz/fdtd_tmz_backend.h"
#include "fdtd/TMz/fdtd_tmz_boundary.h"
#include "fdtd/TMz/fdtd_tmz_simulation.h"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/** Namespace for the fdtd library. */
namespace fdtd
{
    /** One option from the command line, with its value. */
    struct Argument
    {
        /** Name as written, such as "--nx" or "-h". */
        std::string_view name;

        /** Value, empty for a flag. */
        std::string_view value;
    };

    /**
     * Split the command-line arguments, without the program name, into options.
     *
     * An option takes its value as the next argument or after '='; a flag takes none.
     *
     * @param args Arguments in order.
     * @param flags Names of the options that take no value.
     * @return The options in order.
     * @throws std::invalid_argument On an argument that is not an option, a missing
     *         value, or a flag given a value.
     */
    std::vector<Argument> split_arguments(std::span<const std::string_view> args,
                                          std::span<const std::string_view> flags);

    /**
     * Parse a whole non-negative integer.
     *
     * @param name Option name for the error message.
     * @param text Value to parse.
     * @return The number.
     * @throws std::invalid_argument When the text is not a whole number.
     */
    std::size_t parse_size_arg(std::string_view name, std::string_view text);

    /**
     * Parse a positive finite number.
     *
     * @param name Option name for the error message.
     * @param text Value to parse.
     * @return The number.
     * @throws std::invalid_argument When the text is not a positive finite number.
     */
    double parse_positive_arg(std::string_view name, std::string_view text);

    /**
     * Parse a non-negative finite number.
     *
     * @param name Option name for the error message.
     * @param text Value to parse.
     * @return The number.
     * @throws std::invalid_argument When the text is not a non-negative finite number.
     */
    double parse_nonnegative_arg(std::string_view name, std::string_view text);

    /**
     * Split a comma-separated list into its fields.
     *
     * @param text List to split.
     * @return The fields, empty ones included.
     */
    std::vector<std::string_view> split_list_arg(std::string_view text);

    /**
     * Parse a boundary name: pec, pmc or pml.
     *
     * @param name Option name for the error message.
     * @param text Value to parse.
     * @return The boundary.
     * @throws std::invalid_argument On another name.
     */
    TMz::Boundary parse_boundary_arg(std::string_view name, std::string_view text);

    /**
     * Parse a backend name: serial, openmp or cuda, whether compiled in or not.
     *
     * @param name Option name for the error message.
     * @param text Value to parse.
     * @return The backend type.
     * @throws std::invalid_argument On another name.
     */
    TMz::BackendType parse_backend_arg(std::string_view name, std::string_view text);

    /**
     * Parse an OpenMP variant name, `<schedule>[:<W>x<H>][+simd][+phase]`, into omp_ fields.
     *
     * The schedule is `rows`, `collapse2`, `tiles` or `tasks`; `:WxH` sets the tile size,
     * for `tiles` and `tasks` only; `+simd` and `+phase` turn on SIMD and a region per
     * phase, each once, in either order. `default` names the default settings. The thread
     * count is not part of the name: omp_threads is left as it is.
     *
     * @param name Option name for the error message.
     * @param text Variant name, as `tiles:64x16+simd`.
     * @param options Options whose omp_ fields are set; unchanged on an error.
     * @throws std::invalid_argument When the name is malformed.
     */
    void parse_omp_variant_arg(std::string_view name, std::string_view text,
                               TMz::Simulation::Options &options);

    /**
     * Canonical OpenMP variant name of simulation options, which parse_omp_variant_arg() reads
     * back.
     *
     * The tile size is written for `tiles` and `tasks` only, then `+simd`, then `+phase`;
     * the thread count is left out.
     *
     * @param options Options to name.
     * @return The name, as `tasks:128x32+simd`.
     */
    std::string omp_variant_name(const TMz::Simulation::Options &options);

    /**
     * Parse a CUDA variant name, `<kernel>[:<BX>x<BY>]`, into cuda_ fields.
     *
     * The kernel is `fused` or `split`; `:BXxBY` sets the thread block, the default one when it
     * is left out. `default` names the default settings. Whether the device takes the block is
     * left to the backend.
     *
     * @param name Option name for the error message.
     * @param text Variant name, as `split:16x16`.
     * @param options Options whose cuda_ fields are set; unchanged on an error.
     * @throws std::invalid_argument When the name is malformed.
     */
    void parse_cuda_variant_arg(std::string_view name, std::string_view text,
                                TMz::Simulation::Options &options);

    /**
     * Canonical CUDA variant name of simulation options, which parse_cuda_variant_arg() reads
     * back.
     *
     * The block is always written.
     *
     * @param options Options to name.
     * @return The name, as `fused:32x8`.
     */
    std::string cuda_variant_name(const TMz::Simulation::Options &options);

    /**
     * Whether a backend reads a variant name.
     *
     * Every backend reads `default`, the only name serial reads; OpenMP reads the names of
     * parse_omp_variant_arg() and CUDA those of parse_cuda_variant_arg().
     *
     * @param backend Backend type.
     * @param text Variant name.
     * @return True when parse_variant_arg() takes the name for the backend.
     */
    bool backend_reads_variant(TMz::BackendType backend, std::string_view text);

    /**
     * Parse a variant name for a backend into its fields.
     *
     * @param name Option name for the error message.
     * @param backend Backend whose grammar reads the name.
     * @param text Variant name.
     * @param options Options whose fields of the backend are set; unchanged on an error.
     * @throws std::invalid_argument When the backend does not read the name.
     */
    void parse_variant_arg(std::string_view name, TMz::BackendType backend, std::string_view text,
                           TMz::Simulation::Options &options);

    /**
     * Canonical variant name of simulation options for a backend.
     *
     * @param backend Backend type.
     * @param options Options to name.
     * @return "default" for serial, omp_variant_name() for OpenMP, cuda_variant_name() for CUDA.
     */
    std::string variant_name(TMz::BackendType backend, const TMz::Simulation::Options &options);
}

#endif
