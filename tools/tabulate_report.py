#!/usr/bin/env python3
#
# Write the Markdown performance report of a benchmark run and a profile run.
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

import os
import argparse
import string
import datetime
import numpy as np

TEMPLATES = os.path.join(os.path.dirname(__file__), "report")

CONFIG = ["backend", "variant", "precision", "threads", "nx", "ny", "nsteps"]

FIGURES = [
    ("throughput", "Throughput versus grid size"),
    ("speedup", "Speedup versus number of threads"),
    ("efficiency", "Efficiency versus number of threads"),
    ("weak_efficiency", "Weak scaling efficiency versus number of threads"),
]

MEMCHECK = [
    "ERROR SUMMARY",
    "in use at exit",
    "total heap usage",
    "All heap blocks",
    "definitely lost",
    "indirectly lost",
    "possibly lost",
    "still reachable",
]

TOP = 10


def load_info(path):
    """Read the '# key: value' lines of a file as a dict."""
    info = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                info[key.strip()] = value.strip()

    return info


def load_samples(path):
    """Read the samples of a benchmark CSV, skipping the '#' lines."""
    with open(path, encoding="utf-8") as f:
        rows = [line for line in f if not line.startswith("#")]

    data = np.genfromtxt(
        rows,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    return np.atleast_1d(data)


def fill(name, **values):
    """Return the template report/<name> with the values put in."""
    with open(os.path.join(TEMPLATES, name), encoding="utf-8") as f:
        return string.Template(f.read()).substitute(values).strip()


def run_date(stamp):
    """Return a timestamp in milliseconds as 'January 19, 2026 (14:05:09Z)', in UTC."""
    start = datetime.datetime.fromtimestamp(int(stamp) / 1000, tz=datetime.timezone.utc)
    return f"{start:%B} {start.day}, {start:%Y} ({start:%H:%M:%S}Z)"


def table(titles, align, rows):
    """Return a Markdown table; align holds an 'l' or an 'r' per column."""
    lines = [
        "| " + " | ".join(titles) + " |",
        "|" + "|".join(":--" if a == "l" else "--:" for a in align) + "|",
    ]

    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    return "\n".join(lines)


def function(name):
    """Return a function name as code, cut so the tables fit the page."""
    if len(name) > 60:
        name = name[:57] + "..."
    return f"`{name}`"


def sources_table(runs):
    """Return the runs of the report, their start and their commit, as a table."""
    rows = []

    for label, run in runs:
        commit = run["info"].get("git_commit", "unknown")
        rows.append([label, f"`{run['stamp']}`", run_date(run["stamp"]), commit])

    return table(["Run", "Timestamp", "Started", "Commit"], "llll", rows)


def machine_table(runs):
    """Return the machine description of each run as a table, a column per run."""
    keys = []
    for _, run in runs:
        for key in run["info"]:
            if key not in keys and not key.startswith(("profile_", "skipped_")):
                keys.append(key)

    rows = []
    for key in keys:
        rows.append([f"`{key}`"] + [run["info"].get(key, "") for _, run in runs])

    return table(["Key"] + [label for label, _ in runs], "l" * (len(runs) + 1), rows)


def results_table(data):
    """Return the median time, throughput and checksum of each configuration as a table."""
    rows = []

    for config in np.unique(data[CONFIG]):
        samples = data[data[CONFIG] == config]
        backend, variant, precision, threads, nx, ny, nsteps = config

        median = np.median(samples["seconds"])
        throughput = nx * ny * nsteps / median / 1e6

        checksums = np.unique(samples["checksum"])
        if len(checksums) == 1:
            checksum = f"{checksums[0]:.17g}"
        else:
            checksum = f"{len(checksums)} differ"

        rows.append([
            backend,
            variant,
            precision,
            threads,
            f"{nx} x {ny}",
            nsteps,
            len(samples),
            f"{median:.6f}",
            f"{throughput:.1f}",
            checksum,
        ])

    return table(
        ["Backend", "Variant", "Precision", "Threads", "Grid", "Steps", "Samples", "Median (s)",
         "Mcells/s", "Checksum"],
        "lllrrrrrrr",
        rows,
    )


def figure_list(report, stamp):
    """Return the Markdown of the figures plot_benchmark.py drew into REPORT/images."""
    lines = []

    for name, caption in FIGURES:
        image = f"images/{name}_{stamp}.png"
        if os.path.exists(os.path.join(report, image)):
            lines.append(f"![{caption}]({image}){{width=90%}}")

    if not os.path.exists(os.path.join(report, f"images/speedup_{stamp}.png")):
        lines.append("Speedup and efficiency need OpenMP and serial samples of the same size.")

    return "\n\n".join(lines)


def perf_stat(path):
    """Return the counters of a 'perf stat -x,' file as a table."""
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split(",")
            if len(fields) >= 3 and not line.startswith("#"):
                rows.append([f"`{fields[2]}`", fields[0], fields[1]])

    return table(["Event", "Count", "Unit"], "lrl", rows)


def perf_record(path):
    """Return the top functions of a 'perf report --stdio' file as a table."""
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.split(maxsplit=2)
            if len(fields) == 3 and fields[0].endswith("%") and len(rows) < TOP:
                rows.append([fields[0], function(fields[2].strip())])

    return table(["Samples", "Function"], "rl", rows)


def callgrind(path):
    """Return the functions of a callgrind file with the most instructions of their own.

    The file is written with --compress-strings=no --compress-pos=no, so a 'fn=' line names the
    function. The cost line after a 'calls=' line is the cost of a call, not the function's own.
    """
    own = {}
    name = None
    call = False

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("fn="):
                name = line[3:].strip()
            elif line.startswith("calls="):
                call = True
            elif line[:1].isdigit():
                fields = line.split()
                if not call and len(fields) > 1:
                    own[name] = own.get(name, 0) + int(fields[1])
                call = False

    total = sum(own.values())
    rows = []
    for name in sorted(own, key=own.get, reverse=True)[:TOP]:
        rows.append([f"{own[name]:,}", f"{100 * own[name] / total:.1f}%", function(name)])

    return table(["Instructions", "Share", "Function"], "rrl", rows)


def cachegrind(path):
    """Return the miss rates of the caches a cachegrind file simulated, from its summary."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("events:"):
                events = line.split()[1:]
            elif line.startswith("summary:"):
                count = dict(zip(events, map(int, line.split()[1:])))

    data = count["Dr"] + count["Dw"]
    rows = []
    for cache, accesses, misses in [
        ("I1", count["Ir"], count["I1mr"]),
        ("D1", data, count["D1mr"] + count["D1mw"]),
        ("LL instructions", count["Ir"], count["ILmr"]),
        ("LL data", data, count["DLmr"] + count["DLmw"]),
    ]:
        rows.append([cache, f"{accesses:,}", f"{misses:,}", f"{100 * misses / accesses:.2f}%"])

    return table(["Cache", "Accesses", "Misses", "Miss rate"], "lrrr", rows)


def memcheck(path):
    """Return the error and heap summary lines of a memcheck log as a code block."""
    lines = []

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.split("==", 2)[-1].strip()
            if text.startswith(tuple(MEMCHECK)):
                lines.append(text)

    return "```\n" + "\n".join(lines) + "\n```"


def tool(profile, name, suffix, summarize):
    """Return the summary of a tool's output file in a profile run, or why there is none."""
    path = os.path.join(profile["dir"], f"profile_{name}_{profile['stamp']}.{suffix}")

    if os.path.exists(path):
        return summarize(path)

    return "Skipped: " + profile["info"].get(f"skipped_{name}", "not in PROFILE_TOOLS") + "."


def benchmark_section(bench, report, stamp):
    """Return the Benchmark section of the report."""
    if bench is None:
        return "No benchmark run."

    data = bench["samples"]

    weak = "No weak scaling run."
    if bench["weak"] is not None:
        weak = results_table(bench["weak"])

    return fill(
        "benchmark_section.md",
        stamp=bench["stamp"],
        samples=len(data),
        configurations=len(np.unique(data[CONFIG])),
        results=results_table(data),
        weak=weak,
        figures=figure_list(report, stamp),
    )


def profile_section(profile):
    """Return the Profile section of the report."""
    if profile is None:
        return "No profile run."

    info = profile["info"]
    return fill(
        "profile_section.md",
        stamp=profile["stamp"],
        backend=info["profile_backend"],
        threads=info["profile_threads"],
        perf_size=info["profile_perf_size"],
        perf_steps=info["profile_perf_steps"],
        valgrind_size=info["profile_valgrind_size"],
        valgrind_steps=info["profile_valgrind_steps"],
        perf_stat=tool(profile, "perf-stat", "csv", perf_stat),
        perf_record=tool(profile, "perf-record", "txt", perf_record),
        callgrind=tool(profile, "callgrind", "out", callgrind),
        cachegrind=tool(profile, "cachegrind", "out", cachegrind),
        memcheck=tool(profile, "memcheck", "txt", memcheck),
    )


def main():
    """Parse the arguments and write the report."""
    parser = argparse.ArgumentParser(
        description="Write the Markdown report of a benchmark run and a profile run into REPORT."
    )

    parser.add_argument("report", help="report directory, named after its timestamp")
    parser.add_argument("-b", "--benchmark", help="benchmark run directory")
    parser.add_argument("-p", "--profile", help="profile run directory")

    args = parser.parse_args()

    runs = []
    bench = None
    profile = None

    if args.benchmark:
        stamp = os.path.basename(os.path.normpath(args.benchmark))
        path = os.path.join(args.benchmark, f"benchmark_{stamp}.csv")
        weak = os.path.join(args.benchmark, f"benchmark_weak_{stamp}.csv")
        bench = {
            "stamp": stamp,
            "info": load_info(path),
            "samples": load_samples(path),
            "weak": load_samples(weak) if os.path.exists(weak) else None,
        }
        runs.append(("Benchmark", bench))

    if args.profile:
        stamp = os.path.basename(os.path.normpath(args.profile))
        path = os.path.join(args.profile, f"profile_{stamp}.txt")
        profile = {"dir": args.profile, "stamp": stamp, "info": load_info(path)}
        runs.append(("Profile", profile))

    if not runs:
        parser.error("give a benchmark run (-b), a profile run (-p) or both")

    stamp = os.path.basename(os.path.normpath(args.report))
    text = fill(
        "performance_report.md",
        date=run_date(stamp),
        stamp=stamp,
        sources=sources_table(runs),
        machine=machine_table(runs),
        benchmark=benchmark_section(bench, args.report, stamp),
        profile=profile_section(profile),
    )

    path = os.path.join(args.report, f"report_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("==> Creating report:", path)


if __name__ == "__main__":
    main()
