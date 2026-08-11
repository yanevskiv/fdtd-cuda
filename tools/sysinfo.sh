#!/bin/bash
#
# Print the machine and build description of a benchmark run as '# key: value' lines.
#
# Copyright (C) 2026 Ivan Janevski
#
# fdtd-cuda  is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free  Software  Foundation;  either  version 3 of the License, or (at
# your option) any later version.
#
# fdtd-cuda  is  distributed  in  the  hope that it will be useful, but
# WITHOUT   ANY   WARRANTY;   without  even  the  implied  warranty  of
# MERCHANTABILITY  or  FITNESS  FOR  A  PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You  should  have  received  a copy of the GNU General Public License
# along with fdtd-cuda; if not, see <https://www.gnu.org/licenses/>.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-}"
CACHE="${BUILD_DIR:+$BUILD_DIR/CMakeCache.txt}"

# Print usage instructions
usage() {
    echo "Usage: $0 [BUILD_DIR]" >&2
}

# Print one '# key: value' line, with 'unknown' for an empty value
emit() {
    local value="${2:-}"
    value="${value//$'\n'/ }"
    printf '# %s: %s\n' "$1" "${value:-unknown}"
}

# Read a field from lscpu
lscpu_field() {
    command -v lscpu >/dev/null || return 0
    LC_ALL=C lscpu | sed -n "s/^$1:[[:space:]]*//p" | head -n 1
}

# Read a variable from the CMake cache of the build directory
cache_value() {
    [ -n "$CACHE" ] && [ -r "$CACHE" ] || return 0
    sed -n "s/^$1:[A-Z]*=//p" "$CACHE" | head -n 1
}

# Print the first line a command writes, or nothing when it is missing or fails
first_line() {
    command -v "$1" >/dev/null || return 0
    "$@" 2>/dev/null | head -n 1
}

# Print the date and time of the run in UTC
get_date() {
    emit date "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# Print the commit and whether tracked files differ from it
get_git() {
    local commit="" dirty=""
    if command -v git >/dev/null && git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
        commit="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null)"
        if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
            dirty=yes
        else
            dirty=no
        fi
    fi
    emit git_commit "${commit:-${FDTD_GIT_COMMIT:-}}"
    emit git_dirty "${dirty:-${FDTD_GIT_DIRTY:-}}"
}

# Print the CPU model and its sockets, cores and hardware threads
get_cpu() {
    local sockets cores
    sockets="$(lscpu_field 'Socket(s)')"
    cores="$(lscpu_field 'Core(s) per socket')"
    emit cpu_model "$(lscpu_field 'Model name')"
    emit cpu_sockets "$sockets"
    if [ -n "$sockets" ] && [ -n "$cores" ]; then
        emit cpu_cores "$((sockets * cores))"
    else
        emit cpu_cores ""
    fi
    emit cpu_threads "$(lscpu_field 'CPU(s)')"
}

# Print the total memory in GiB
get_memory() {
    local kib=""
    [ -r /proc/meminfo ] && kib="$(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\) kB$/\1/p' /proc/meminfo)"
    if [ -n "$kib" ]; then
        emit memory "$(awk -v k="$kib" 'BEGIN { printf "%.1f GiB", k / 1048576 }')"
    else
        emit memory ""
    fi
}

# Print the CPU frequency governor
get_governor() {
    local file=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    if [ -r "$file" ]; then
        emit cpu_governor "$(cat "$file")"
    else
        emit cpu_governor ""
    fi
}

# Print the name, memory and driver of the first GPU
get_gpu() {
    local line="" name="" memory="" driver=""
    line="$(first_line nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader)"
    if [ -n "$line" ]; then
        IFS=',' read -r name memory driver <<< "$line"
    fi
    emit gpu_name "${name# }"
    emit gpu_memory "${memory# }"
    emit gpu_driver "${driver# }"
}

# Print the C++ compiler of the build and the nvcc version
get_compilers() {
    local cxx nvcc
    cxx="$(cache_value CMAKE_CXX_COMPILER)"
    emit cxx_compiler "${cxx:+$(first_line "$cxx" --version)}"
    nvcc="$(cache_value CMAKE_CUDA_COMPILER)"
    if [ -z "$nvcc" ] || [ "$nvcc" = NOTFOUND ] || [ "${nvcc%-NOTFOUND}" != "$nvcc" ]; then
        nvcc=nvcc
    fi
    emit nvcc_version "$(command -v "$nvcc" >/dev/null && "$nvcc" --version 2>/dev/null \
        | sed -n 's/^Cuda compilation tools, //p')"
}

# Print the build type, the C++ flags and the fdtd options of the build
get_cmake() {
    local type flags
    type="$(cache_value CMAKE_BUILD_TYPE)"
    flags="$(cache_value CMAKE_CXX_FLAGS)"
    if [ -n "$type" ]; then
        flags="$flags $(cache_value "CMAKE_CXX_FLAGS_${type^^}")"
    fi
    flags="${flags# }"
    emit build_type "$type"
    emit cxx_flags "${CACHE:+${flags:-none}}"
    emit fdtd_double "$(cache_value FDTD_DOUBLE)"
    emit fdtd_with_openmp "$(cache_value FDTD_WITH_OPENMP)"
    emit fdtd_with_cuda "$(cache_value FDTD_WITH_CUDA)"
}

if [ "$#" -gt 1 ] || [ "${1:-}" = -h ] || [ "${1:-}" = --help ]; then
    usage
    [ "$#" -gt 1 ] && exit 2
    exit 0
fi

get_date
get_git
get_cpu
get_memory
get_governor
get_gpu
get_compilers
get_cmake
exit 0
