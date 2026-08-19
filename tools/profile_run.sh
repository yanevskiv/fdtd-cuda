#!/bin/bash
#
# Run the benchmark program under the profilers of the project's config file, one run directory.
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

set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(dirname "$HERE")/config"
NAME="../config"
PROG=""
OUT="$(dirname "$HERE")/results/profiles"
TOOLS="perf-stat perf-record callgrind cachegrind memcheck nsys ncu"
PERF_EVENTS="task-clock,cycles:u,instructions:u,branches:u,branch-misses:u"
PERF_EVENTS="$PERF_EVENTS,cache-references:u,cache-misses:u"
NCU_SKIP=20
NCU_COUNT=10

# Print usage instructions
usage() {
    echo "Usage: $0 -p PROGRAM [-c CONFIG] [-o OUTDIR]" >&2
}

# Print an error and the usage, then exit 2
usage_error() {
    echo "profile_run.sh: $1" >&2
    usage
    exit 2
}

# Print an error about the config file and exit 1
config_error() {
    echo "profile_run.sh: $1 (try: cp config.example config)" >&2
    exit 1
}

# Print a line on standard error and append it to the run log
log() {
    echo "$1" | tee -a "$LOG" >&2
}

# Append one '# key: value' line to the run's description
describe() {
    printf '# %s: %s\n' "$1" "$2" >> "$INFO"
}

# Record a tool as skipped, with the reason, in the description and the log
skip() {
    describe "skipped_$1" "$2"
    log "==> $1: skipped, $2"
}

# Run a command with its standard output dropped and its standard error logged; exit 1 on failure
run() {
    "$@" 2>&1 >/dev/null | tee -a "$LOG" >&2 || exit 1
}

# Print why perf cannot count or sample this program, or nothing when it can
perf_refusal() {
    local paranoid
    command -v perf >/dev/null || { echo "perf is missing"; return; }
    perf stat -e instructions:u -- true >/dev/null 2>&1 && return
    paranoid="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)"
    echo "perf refused (perf_event_paranoid=${paranoid:-unknown})"
}

# Print why valgrind cannot run, or nothing when it can
valgrind_refusal() {
    command -v valgrind >/dev/null || echo "valgrind is missing"
}

# Print why a GPU tool cannot profile this run, or nothing when it can
gpu_refusal() {
    command -v "$1" >/dev/null || { echo "$1 is missing"; return; }
    [ "$PROFILE_BACKEND" = cuda ] || echo "$1 needs PROFILE_BACKEND=cuda"
}

# Print why ncu cannot read the GPU counters, or nothing when it can
ncu_refusal() {
    local params=/proc/driver/nvidia/params reason
    reason="$(gpu_refusal ncu)"
    [ -z "$reason" ] || { echo "$reason"; return; }
    [ "$(id -u)" -ne 0 ] && [ -r "$params" ] && grep -q '^RmProfilingAdminOnly: 1' "$params" \
        && echo "ncu refused (RmProfilingAdminOnly=1, counters for admins only)"
}

# Count hardware events over the whole run with perf stat
run_perf_stat() {
    run perf stat -x, -o "$DIR/profile_perf-stat_$STAMP.csv" -e "$PERF_EVENTS" \
        -- "$PROG" "${PERF_ARGS[@]}"
}

# Sample the run with perf record, with frame-pointer call graphs, and save the report as text
run_perf_record() {
    local data="$DIR/profile_perf-record_$STAMP.data"
    run perf record -q --call-graph fp -e cycles:u -o "$data" -- "$PROG" "${PERF_ARGS[@]}"
    perf report --stdio --no-children --sort symbol -g none --percent-limit 0.1 -i "$data" \
        > "$DIR/profile_perf-record_$STAMP.txt" 2>> "$LOG" || exit 1
}

# Count instructions per function and call with callgrind, and save callgrind_annotate's text
run_callgrind() {
    local out="$DIR/profile_callgrind_$STAMP.out"
    run valgrind -q --tool=callgrind --callgrind-out-file="$out" \
        --compress-strings=no --compress-pos=no -- "$PROG" "${VALGRIND_ARGS[@]}"
    if command -v callgrind_annotate >/dev/null; then
        callgrind_annotate --inclusive=yes "$out" > "$DIR/profile_callgrind_$STAMP.txt" \
            2>> "$LOG" || exit 1
    fi
}

# Simulate the caches with cachegrind, and save cg_annotate's text
run_cachegrind() {
    local out="$DIR/profile_cachegrind_$STAMP.out"
    run valgrind -q --tool=cachegrind --cache-sim=yes --cachegrind-out-file="$out" \
        -- "$PROG" "${VALGRIND_ARGS[@]}"
    if command -v cg_annotate >/dev/null; then
        cg_annotate "$out" > "$DIR/profile_cachegrind_$STAMP.txt" 2>> "$LOG" || exit 1
    fi
}

# Check memory accesses and leaks with memcheck, its report into a text file
run_memcheck() {
    run valgrind --tool=memcheck --leak-check=full \
        --log-file="$DIR/profile_memcheck_$STAMP.txt" -- "$PROG" "${VALGRIND_ARGS[@]}"
}

# Trace the CUDA calls and kernels with nsys, and save its kernel and API summaries as CSV
run_nsys() {
    local base="$DIR/profile_nsys_$STAMP"
    run nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
        -o "$base" "$PROG" "${PERF_ARGS[@]}"
    run nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --format csv \
        --output "$base" "$base.nsys-rep"
    mv "${base}_cuda_gpu_kern_sum.csv" "$base.csv" || exit 1
    mv "${base}_cuda_api_sum.csv" "$DIR/profile_nsys-api_$STAMP.csv" || exit 1
}

# Measure the speed of light of a few kernels with ncu, and save its details as CSV
run_ncu() {
    local base="$DIR/profile_ncu_$STAMP"
    run ncu --section SpeedOfLight --launch-skip "$NCU_SKIP" --launch-count "$NCU_COUNT" \
        --export "$base" --force-overwrite "$PROG" "${PERF_ARGS[@]}"
    ncu --import "$base.ncu-rep" --csv --page details > "$base.csv" 2>> "$LOG" || exit 1
}

while getopts "p:c:o:h" opt; do
    case "$opt" in
        p) PROG="$OPTARG" ;;
        c) CONFIG="$OPTARG"; NAME="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

[ -n "$PROG" ] && [ -x "$PROG" ] || usage_error "-p must name the benchmark program"
[ "$#" -eq 0 ] || usage_error "no arguments after the options; the settings are in $NAME"
[ -f "$CONFIG" ] || config_error "can't find '$NAME' file"

# shellcheck source=../config.example
. "$CONFIG" || config_error "can't load '$NAME' file"
for name in PROFILE_TOOLS PROFILE_BACKEND PROFILE_THREADS PROFILE_PERF_SIZE PROFILE_PERF_STEPS \
    PROFILE_VALGRIND_SIZE PROFILE_VALGRIND_STEPS; do
    [ -n "${!name:-}" ] || config_error "'$NAME' file doesn't set $name"
done
for tool in $PROFILE_TOOLS; do
    [[ " $TOOLS " == *" $tool "* ]] || config_error "'$NAME' file names the unknown tool $tool"
done

# One sample with no warm-up, so the profile holds a single run of the timed code.
COMMON=(--repeat 1 --warmup 0 --backend "$PROFILE_BACKEND" --threads "$PROFILE_THREADS")
PERF_ARGS=(--nx "$PROFILE_PERF_SIZE" --steps "$PROFILE_PERF_STEPS" "${COMMON[@]}")
VALGRIND_ARGS=(--nx "$PROFILE_VALGRIND_SIZE" --steps "$PROFILE_VALGRIND_STEPS" "${COMMON[@]}")

[ "${OUT#/}" != "$OUT" ] || OUT="$PWD/$OUT"

# A run is named after its start in milliseconds since 1970, in its directory and every file.
STAMP="$(date +%s%3N)"
DIR="$OUT/$STAMP"
INFO="$DIR/profile_$STAMP.txt"
LOG="$DIR/profile_$STAMP.log"
if [ -e "$DIR" ]; then
    echo "profile_run.sh: $DIR already exists" >&2
    exit 1
fi
mkdir -p "$DIR" || exit 1
: > "$LOG"

# The description holds the machine, then the settings, then a line per skipped tool.
"$HERE/sysinfo.sh" "$(dirname "$PROG")" > "$INFO"
describe profile_tools "$PROFILE_TOOLS"
describe profile_backend "$PROFILE_BACKEND"
describe profile_threads "$PROFILE_THREADS"
describe profile_perf_size "$PROFILE_PERF_SIZE"
describe profile_perf_steps "$PROFILE_PERF_STEPS"
describe profile_valgrind_size "$PROFILE_VALGRIND_SIZE"
describe profile_valgrind_steps "$PROFILE_VALGRIND_STEPS"
describe profile_ncu_skip "$NCU_SKIP"
describe profile_ncu_count "$NCU_COUNT"

log "==> config $CONFIG: tools=($PROFILE_TOOLS) backend=$PROFILE_BACKEND\
 threads=$PROFILE_THREADS perf=${PROFILE_PERF_SIZE}^2x$PROFILE_PERF_STEPS\
 valgrind=${PROFILE_VALGRIND_SIZE}^2x$PROFILE_VALGRIND_STEPS"

PERF="$(perf_refusal)"
VALGRIND="$(valgrind_refusal)"
NSYS="$(gpu_refusal nsys)"
NCU="$(ncu_refusal)"
for tool in $PROFILE_TOOLS; do
    case "$tool" in
        perf-*) reason="$PERF" ;;
        nsys) reason="$NSYS" ;;
        ncu) reason="$NCU" ;;
        *) reason="$VALGRIND" ;;
    esac
    if [ -n "$reason" ]; then
        skip "$tool" "$reason"
        continue
    fi
    log "==> $tool"
    "run_${tool//-/_}"
done
log "==> wrote $DIR"
