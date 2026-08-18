#!/usr/bin/env python3
#
# Write the Markdown performance report of benchmark runs and a profile run.
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
import csv
import argparse
import string
import datetime
import numpy as np

TEMPLATES = os.path.join(os.path.dirname(__file__), "report")

CONFIG = ["backend", "variant", "precision", "threads", "nx", "ny", "nsteps", "label"]

FIGURES = [
    ("throughput", "Throughput versus grid size"),
    ("speedup_size", "Speedup over serial versus grid size"),
    ("bandwidth", "Achieved memory bandwidth versus grid size"),
    ("roofline", "Roofline at the largest grid size"),
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

NCU_METRICS = [
    "Duration",
    "Memory Throughput",
    "DRAM Throughput",
    "Compute (SM) Throughput",
]

MICROSECONDS = {
    "nsecond": 1e-3,
    "usecond": 1.0,
    "msecond": 1e3,
    "second": 1e6,
}


def load_info(path):
    """Read the '# key: value' lines of a file as a dict."""
    info = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                info[key.strip()] = value.strip()

    return info


def load_samples(runs):
    """Read the samples of the CSVs of the runs, a (path, label) pair each, into one array.

    The label of each run goes into the column 'label'. Columns are only ever added at the end
    of the CSV, so a run that lacks the last ones gets nan in them.
    """
    header = []
    files = []
    for path, label in runs:
        with open(path, encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if not line.startswith("#")]

        columns = lines[0].split(",")
        if len(columns) > len(header):
            header = columns
        files.append((label, lines[1:]))

    rows = [",".join(header + ["label"])]
    for label, lines in files:
        for line in lines:
            values = line.split(",")
            values += ["nan"] * (len(header) - len(values))
            rows.append(",".join(values + [label]))

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


def kernel(name):
    """Return a CUDA kernel's name without its namespaces and parameters, as code."""
    name = name.split("(")[0]
    return f"`{name.split('::')[-1]}`"


def sources_table(runs):
    """Return the runs of the report, their start and their commit, as a table."""
    rows = []

    for label, run in runs:
        commit = run["info"].get("git_commit", "unknown")
        rows.append([label, f"`{run['stamp']}`", run_date(run["stamp"]), commit])

    return table(["Run", "Timestamp", "Started", "Commit"], "llll", rows)


def machine_table(runs):
    """
    Return the machine description of the runs as tables.

    Returns the keys every run gives the same value in one table, a value each, then the keys
    the runs differ in, a column per run, so long values such as the CPU model fit the page.
    """
    keys = []
    for _, run in runs:
        for key in run["info"]:
            if key not in keys and not key.startswith(("profile_", "skipped_")):
                keys.append(key)

    shared = []
    differ = []
    for key in keys:
        values = [run["info"].get(key, "") for _, run in runs]
        if len(set(values)) == 1:
            shared.append([f"`{key}`", values[0]])
        else:
            differ.append([f"`{key}`"] + values)

    text = table(["Key", "Value"], "ll", shared)
    if differ:
        text += "\n\nThe keys the runs differ in:\n\n"
        text += table(["Key"] + [label for label, _ in runs], "l" * (len(runs) + 1), differ)

    return text


def results_table(data):
    """
    Return the median time, its spread, the throughput and the checksum of each configuration
    as a table.

    The spread is the interquartile range as a percent of the median. A column every row holds
    the same value in, the precision, the steps or the samples, is left out of the table and
    given in a line above it, so the table fits the page; the run label only shows when the runs
    have more than one.
    """
    columns = [
        ("Run", "label", "l"),
        ("Precision", "precision", "l"),
        ("Steps", "nsteps", "r"),
        ("Samples", None, "r"),
    ]

    configs = np.unique(data[CONFIG])
    counts = [np.sum(data[CONFIG] == config) for config in configs]

    common = []
    shown = []
    for title, name, align in columns:
        values = counts if name is None else [config[name] for config in configs]
        if len(set(values)) == 1:
            if name != "label":
                common.append(f"{title.lower()} {values[0]}")
        else:
            shown.append((title, name, align))

    rows = []
    for config, count in zip(configs, counts):
        samples = data[data[CONFIG] == config]
        nx = config["nx"]
        ny = config["ny"]

        median = np.median(samples["seconds"])
        spread = np.percentile(samples["seconds"], 75) - np.percentile(samples["seconds"], 25)
        throughput = nx * ny * config["nsteps"] / median / 1e6

        checksums = np.unique(samples["checksum"])
        if len(checksums) == 1:
            checksum = f"{checksums[0]:.17g}"
        else:
            checksum = f"{len(checksums)} differ"

        row = [config["backend"], config["variant"]]
        for _, name, _ in shown:
            row.append(count if name is None else config[name])
        row += [
            config["threads"],
            f"{nx} x {ny}",
            f"{median:.6f}",
            f"{100 * spread / median:.1f}",
            f"{throughput:.1f}",
            checksum,
        ]
        rows.append(row)

    titles = ["Backend", "Variant"] + [title for title, _, _ in shown]
    titles += ["Threads", "Grid", "Median (s)", "IQR (%)", "Mcells/s", "Checksum"]
    align = "ll" + "".join(a for _, _, a in shown) + "rrrrrr"

    text = ""
    if common:
        text = "Every configuration: " + ", ".join(common) + ".\n\n"

    return text + table(titles, align, rows)


def peak(infos, key):
    """Return the number a peak line of the first run that has it starts with, or None."""
    for info in infos:
        if key in info:
            try:
                return float(info[key].split()[0])
            except (IndexError, ValueError):
                return None

    return None


def bandwidth_table(data, infos):
    """
    Return the achieved bandwidth and flops of each configuration at its backend's largest grid,
    and the bandwidth as a percent of the peak the runs measured, as a table.

    The peak of the CUDA backend is the GPU's copy kernel and that of the others the CPU's
    STREAM triad.
    """
    if "bytes_per_step" not in data.dtype.names or np.all(np.isnan(data["bytes_per_step"])):
        return "The runs hold no modelled traffic."

    peaks = {
        "cuda": peak(infos, "gpu_copy_bandwidth"),
        "openmp": peak(infos, "cpu_triad_bandwidth"),
        "serial": peak(infos, "cpu_triad_bandwidth"),
    }

    rows = []
    for config in np.unique(data[CONFIG]):
        backend = config["backend"]
        same = data["backend"] == backend
        if config["nx"] != np.max(data["nx"][same]):
            continue

        samples = data[data[CONFIG] == config]
        median = np.median(samples["seconds"])
        steps = config["nsteps"] / median
        bandwidth = samples["bytes_per_step"][0] * steps / 1e9
        flops = samples["flops_per_step"][0] * steps / 1e9

        fraction = "unknown"
        if peaks.get(backend) is not None:
            fraction = f"{100 * bandwidth / peaks[backend]:.1f}"

        rows.append([
            backend,
            config["variant"],
            config["threads"],
            f"{config['nx']} x {config['ny']}",
            f"{samples['bytes_per_step'][0] / (config['nx'] * config['ny']):.1f}",
            f"{bandwidth:.1f}",
            f"{flops:.1f}",
            fraction,
        ])

    return table(
        ["Backend", "Variant", "Threads", "Grid", "Bytes/cell", "GB/s", "GFLOPS", "Of peak (%)"],
        "llrrrrrr",
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


def nsys(path):
    """Return the GPU kernels of an nsys kernel summary, by total time, as a table."""
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in csv.DictReader(f):
            rows.append([
                kernel(line["Name"]),
                line["Instances"],
                f"{float(line['Total Time (ns)']) / 1e6:.3f}",
                f"{float(line['Avg (ns)']) / 1e3:.2f}",
                f"{float(line['Time (%)']):.1f}%",
            ])

    return table(["Kernel", "Launches", "Total (ms)", "Mean (us)", "Share"], "lrrrr", rows)


def ncu(path):
    """Return the speed-of-light metrics of an ncu details CSV, averaged per kernel, as a table.

    The duration is in microseconds and the throughputs in percent of the peak.
    """
    sums = {}
    launches = {}

    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("==")]

    for line in csv.DictReader(lines):
        metric = line["Metric Name"]
        if metric not in NCU_METRICS:
            continue

        name = kernel(line["Kernel Name"])
        value = float(line["Metric Value"].replace(",", ""))
        if metric == "Duration":
            value *= MICROSECONDS[line["Metric Unit"]]

        sums.setdefault(name, {})
        sums[name][metric] = sums[name].get(metric, 0.0) + value
        launches.setdefault(name, set()).add(line["ID"])

    rows = []
    for name in sums:
        count = len(launches[name])
        row = [name, count]
        for metric in NCU_METRICS:
            row.append(f"{sums[name].get(metric, float('nan')) / count:.1f}")
        rows.append(row)

    return table(
        ["Kernel", "Launches", "Duration (us)", "Memory (%)", "DRAM (%)", "SM (%)"],
        "lrrrrr",
        rows,
    )


def tool(profile, name, suffix, summarize):
    """Return the summary of a tool's output file in a profile run, or why there is none."""
    path = os.path.join(profile["dir"], f"profile_{name}_{profile['stamp']}.{suffix}")

    if os.path.exists(path):
        return summarize(path)

    return "Skipped: " + profile["info"].get(f"skipped_{name}", "not in PROFILE_TOOLS") + "."


def benchmark_section(benches, report, stamp):
    """Return the Benchmark section of the report from its benchmark runs."""
    if not benches:
        return "No benchmark run."

    data = load_samples([(bench["path"], bench["label"]) for bench in benches])

    weak = "No weak scaling run."
    weak_runs = [(bench["weak"], bench["label"]) for bench in benches if bench["weak"]]
    if weak_runs:
        weak = results_table(load_samples(weak_runs))

    infos = [bench["info"] for bench in benches]
    names = [f"`{bench['stamp']}`" for bench in benches]
    if len(names) == 1:
        intro = f"The run {names[0]} holds"
    else:
        intro = "The runs " + ", ".join(names[:-1]) + f" and {names[-1]} hold"

    return fill(
        "benchmark_section.md",
        intro=intro,
        samples=len(data),
        configurations=len(np.unique(data[CONFIG])),
        results=results_table(data),
        bandwidth=bandwidth_table(data, infos),
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
        nsys=tool(profile, "nsys", "csv", nsys),
        ncu=tool(profile, "ncu", "csv", ncu),
        ncu_skip=info.get("profile_ncu_skip", "unknown"),
        ncu_count=info.get("profile_ncu_count", "unknown"),
    )


def main():
    """Parse the arguments and write the report."""
    parser = argparse.ArgumentParser(
        description="Write the Markdown report of a benchmark run and a profile run into REPORT."
    )

    parser.add_argument("report", help="report directory, named after its timestamp")
    parser.add_argument(
        "-b",
        "--benchmark",
        action="append",
        default=[],
        help="benchmark run directory; repeatable",
    )
    parser.add_argument("-p", "--profile", help="profile run directory")

    args = parser.parse_args()

    runs = []
    benches = []
    profile = None

    for directory in args.benchmark:
        stamp = os.path.basename(os.path.normpath(directory))
        path = os.path.join(directory, f"benchmark_{stamp}.csv")
        weak = os.path.join(directory, f"benchmark_weak_{stamp}.csv")
        info = load_info(path)
        benches.append({
            "stamp": stamp,
            "info": info,
            "label": info.get("label", stamp),
            "path": path,
            "weak": weak if os.path.exists(weak) else None,
        })

    for bench in benches:
        name = "Benchmark"
        if len(benches) > 1:
            name += f" {bench['label']}"
        runs.append((name, bench))

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
        benchmark=benchmark_section(benches, args.report, stamp),
        profile=profile_section(profile),
    )

    path = os.path.join(args.report, f"report_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("==> Creating report:", path)


if __name__ == "__main__":
    main()
