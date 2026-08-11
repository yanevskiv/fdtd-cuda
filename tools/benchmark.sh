#!/bin/bash
#
# Run the benchmark program once per grid size and collect the rows into one benchmark.csv.
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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=""
STEPS=500
REPEAT=10
BACKENDS=()
THREADS=()
DIR=""

# Print usage instructions
usage() {
    echo "Usage: $0 -p PROGRAM [-n STEPS] [-r REPEAT] [-b BACKEND]... [-t THREADS]..." \
        "[-o DIR] SIZE..." >&2
}

# Print an error and the usage, then exit 2
usage_error() {
    echo "benchmark.sh: $1" >&2
    usage
    exit 2
}

# Name the default output directory after the date and, in a git checkout, the commit
default_dir() {
    local name commit=""
    name="$(date -u +%Y%m%d-%H%M%S)"
    if command -v git >/dev/null && git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
        commit="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null)"
    fi
    echo "$ROOT/results/benchmarks/$name${commit:+-$commit}"
}

# Print a line on standard error and append it to the run log
log() {
    echo "$1" | tee -a "$LOG" >&2
}

while getopts "p:n:r:b:t:o:h" opt; do
    case "$opt" in
        p) PROG="$OPTARG" ;;
        n) STEPS="$OPTARG" ;;
        r) REPEAT="$OPTARG" ;;
        b) BACKENDS+=(--backend "$OPTARG") ;;
        t) THREADS+=(--threads "$OPTARG") ;;
        o) DIR="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

[ -n "$PROG" ] && [ -x "$PROG" ] || usage_error "-p must name the benchmark program"
[ "$#" -ge 1 ] || usage_error "no grid size given"
DIR="${DIR:-$(default_dir)}"
CSV="$DIR/benchmark.csv"
LOG="$DIR/run.log"
if [ -e "$CSV" ]; then
    echo "benchmark.sh: $CSV already exists" >&2
    exit 1
fi
mkdir -p "$DIR" || exit 1
: > "$LOG"

# The machine description and the CSV header come first, then one block of rows per size.
"$ROOT/tools/sysinfo.sh" "$(dirname "$PROG")" > "$CSV"
"$PROG" --header-only >> "$CSV" || exit 1
for size in "$@"; do
    log "==> nx=ny=$size steps=$STEPS repeat=$REPEAT"
    "$PROG" --nx "$size" --steps "$STEPS" --repeat "$REPEAT" "${BACKENDS[@]}" "${THREADS[@]}" \
        --no-header 2>&1 >> "$CSV" | tee -a "$LOG" >&2 || exit 1
done
log "==> wrote $CSV"
