#!/bin/bash
#
# Run the benchmark sweep of the project's config file and collect the rows into one CSV per run.
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
OUT="$(dirname "$HERE")/results/benchmarks"

# Print usage instructions
usage() {
    echo "Usage: $0 -p PROGRAM [-c CONFIG] [-o OUTDIR]" >&2
}

# Print an error and the usage, then exit 2
usage_error() {
    echo "bench_sweep.sh: $1" >&2
    usage
    exit 2
}

# Print an error about the config file and exit 1
config_error() {
    echo "bench_sweep.sh: $1 (try: cp config.example config)" >&2
    exit 1
}

# Print a line on standard error and append it to the run log
log() {
    echo "$1" | tee -a "$LOG" >&2
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
[ "$#" -eq 0 ] || usage_error "no arguments after the options; the sweep is in $CONFIG"
[ -f "$CONFIG" ] || config_error "can't find '$NAME' file"

# shellcheck source=../config.example
. "$CONFIG" || config_error "can't load '$NAME' file"
for name in BENCH_STEPS BENCH_REPEAT BENCH_BACKENDS BENCH_THREADS BENCH_VARIANTS BENCH_SIZES; do
    [ -n "${!name:-}" ] || config_error "'$NAME' file doesn't set $name"
done

# The settings are space-separated lists, one option of the program per item.
BACKENDS=()
for backend in $BENCH_BACKENDS; do
    BACKENDS+=(--backend "$backend")
done
THREADS=()
for threads in $BENCH_THREADS; do
    THREADS+=(--threads "$threads")
done
VARIANTS=()
for variant in $BENCH_VARIANTS; do
    VARIANTS+=(--variant "$variant")
done
WEAK_NX="${BENCH_WEAK_NX:-}"
ORDER="${BENCH_ORDER:-interleaved}"
BOUNDARY="${BENCH_BOUNDARY:-pml}"
SNAPSHOT_EVERY="${BENCH_SNAPSHOT_EVERY:-0}"
LABEL="${BENCH_LABEL:-}"

[ "${OUT#/}" != "$OUT" ] || OUT="$PWD/$OUT"

# A run is named after its start in milliseconds since 1970, in its directory and every file.
STAMP="$(date +%s%3N)"
DIR="$OUT/$STAMP"
CSV="$DIR/benchmark_$STAMP.csv"
WEAK="$DIR/benchmark_weak_$STAMP.csv"
LOG="$DIR/benchmark_$STAMP.log"
if [ -e "$DIR" ]; then
    echo "bench_sweep.sh: $DIR already exists" >&2
    exit 1
fi
mkdir -p "$DIR" || exit 1
: > "$LOG"

log "==> config $CONFIG: steps=$BENCH_STEPS repeat=$BENCH_REPEAT order=$ORDER\
 boundary=$BOUNDARY snapshot_every=$SNAPSHOT_EVERY backends=($BENCH_BACKENDS) threads=($BENCH_THREADS) variants=($BENCH_VARIANTS)\
 sizes=($BENCH_SIZES) weak_nx=($WEAK_NX) label=$LABEL"

# Print the description of the run: the machine, then the label when one is set
describe() {
    "$HERE/sysinfo.sh" "$(dirname "$PROG")"
    [ -z "$LABEL" ] || printf '# label: %s\n' "$LABEL"
}

# Print the load averages at the end of the run, to set against those at its start
load_at_end() {
    local one five fifteen rest
    read -r one five fifteen rest < /proc/loadavg 2>/dev/null || return 0
    printf '# load_average_end: %s %s %s\n' "$one" "$five" "$fifteen"
}

# The run's description, the machine's peaks and the CSV header come first, then one block of
# rows per size.
describe > "$CSV"
log "==> peaks"
"$PROG" --peaks >> "$CSV" || exit 1
"$PROG" --header-only >> "$CSV" || exit 1
for size in $BENCH_SIZES; do
    log "==> nx=ny=$size steps=$BENCH_STEPS repeat=$BENCH_REPEAT"
    "$PROG" --nx "$size" --steps "$BENCH_STEPS" --repeat "$BENCH_REPEAT" --order "$ORDER" \
        --boundary "$BOUNDARY" --snapshot-every "$SNAPSHOT_EVERY" "${BACKENDS[@]}" "${THREADS[@]}" "${VARIANTS[@]}" --no-header 2>&1 >> "$CSV" \
        | tee -a "$LOG" >&2 || exit 1
done
load_at_end >> "$CSV"
log "==> wrote $CSV"

# Weak scaling keeps the cells per thread fixed: p threads get p rows of nx by nx blocks.
[ -n "$WEAK_NX" ] || exit 0
describe > "$WEAK"
"$PROG" --header-only >> "$WEAK" || exit 1
for threads in $BENCH_THREADS; do
    log "==> weak nx=$WEAK_NX ny=$((WEAK_NX * threads)) threads=$threads"
    "$PROG" --nx "$WEAK_NX" --ny "$((WEAK_NX * threads))" --steps "$BENCH_STEPS" \
        --repeat "$BENCH_REPEAT" --order "$ORDER" --boundary "$BOUNDARY" --backend openmp \
        --threads "$threads" "${VARIANTS[@]}" --no-header 2>&1 >> "$WEAK" \
        | tee -a "$LOG" >&2 || exit 1
done
load_at_end >> "$WEAK"
log "==> wrote $WEAK"
