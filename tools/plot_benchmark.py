#!/usr/bin/env python3
#
# Draw the figures of benchmark runs into a report: throughput, speedup and efficiency.
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
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

LEGEND_INSIDE = 8


def load_info(path):
    """Read the '# key: value' lines of a run's CSV as a dict."""
    info = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                info[key.strip()] = value.strip()

    return info


def peak(infos, key):
    """Return the number a peak line of the first run that has it starts with, or None."""
    for info in infos:
        if key in info:
            try:
                return float(info[key].split()[0])
            except (IndexError, ValueError):
                return None

    return None


def load(runs):
    """Read the samples of the CSVs of the runs, a (path, label) pair each, into one array.

    Where the runs differ in precision or label, the variant name carries both, as
    'v1:32x8 [double / passive]', so that every figure tells the runs apart, and the column
    'run' holds them for the speedup to pair each sample with the serial samples of its own run.
    Columns are only ever added at the end of the CSV, so a run that lacks the last ones gets
    nan in them.
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

    variant = header.index("variant")
    precision = header.index("precision")

    precisions = set()
    labels = set()
    for label, lines in files:
        labels.add(label)
        for line in lines:
            precisions.add(line.split(",")[precision])

    rows = [",".join(header + ["run"])]
    for label, lines in files:
        for line in lines:
            values = line.split(",")
            values += ["nan"] * (len(header) - len(values))

            tags = []
            if len(precisions) > 1:
                tags.append(values[precision])
            if len(labels) > 1:
                tags.append(label)

            run = " / ".join(tags)
            if run:
                values[variant] += f" [{run}]"
            rows.append(",".join(values + [run or "all"]))

    data = np.genfromtxt(
        rows,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    return np.atleast_1d(data)


def calculate(data):
    """Return the throughput, speedup and efficiency of every sample.

    The speedup is over the serial median of the same grid size and run, and only OpenMP and
    CUDA samples have one; the others are NaN. The efficiency is the speedup over the threads.
    """
    throughput = data["nx"] * data["ny"] * data["nsteps"] / data["seconds"] / 1e6
    speedup = np.full(len(data), np.nan)

    for run in np.unique(data["run"]):
        for nx in np.unique(data["nx"]):
            same = (data["run"] == run) & (data["nx"] == nx)
            serial = data[same & (data["backend"] == "serial")]
            parallel = same & (data["backend"] != "serial")
            if len(serial) > 0:
                speedup[parallel] = np.median(serial["seconds"]) / data["seconds"][parallel]

    efficiency = speedup / data["threads"]
    return throughput, speedup, efficiency


def statistics(data, values, x_column):
    """Return the x values, and the median and the lower and upper quartiles of the values at
    each."""
    xs = np.unique(data[x_column])
    ys = []
    lows = []
    highs = []

    for x in xs:
        selected = values[data[x_column] == x]
        ys.append(np.median(selected))
        lows.append(np.percentile(selected, 25))
        highs.append(np.percentile(selected, 75))

    return xs, np.array(ys), np.array(lows), np.array(highs)


def draw(ax, data, values, x_column, i, label):
    """Draw the median of the values against a column, with the quartiles as error bars."""
    x, y, low, high = statistics(data, values, x_column)

    ax.errorbar(
        x,
        y,
        yerr=[y - low, high - y],
        marker=MARKERS[i % len(MARKERS)],
        linewidth=1.8,
        capsize=3,
        label=label,
    )


def save(fig, args, name):
    """Save a figure as images/<name>_<stamp>.png in the report directory and close it."""
    path = os.path.join(args.output_dir, f"{name}_{args.stamp}.png")
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("==> Creating image:", path)


def setup(ax, xlabel, ylabel, title):
    """Apply the labels, title and grid every plot shares."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def log_y(ax):
    """Make the y axis logarithmic, labelled in plain numbers at 1, 2 and 5 per decade."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def legend(ax):
    """Draw the legend inside the axes, or beside them when it holds many curves."""
    count = len(ax.get_legend_handles_labels()[1])
    if count > LEGEND_INSIDE:
        ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        ax.legend()


def plot_by_size(args, data, values):
    """Plot throughput against grid size, with one curve per backend, variant and threads.

    With more than one variant of a backend, only its largest thread count is drawn, so the
    figure stays readable.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    i = 0
    for backend in np.unique(data["backend"]):
        selected = data["backend"] == backend
        variants = np.unique(data["variant"][selected])
        threads = np.unique(data["threads"][selected])
        if len(variants) > 1:
            threads = threads[-1:]

        for variant in variants:
            for p in threads:
                mask = selected & (data["variant"] == variant) & (data["threads"] == p)
                if not np.any(mask):
                    continue

                label = backend
                if len(variants) > 1:
                    label += f" {variant}"
                if len(np.unique(data["threads"][selected])) > 1:
                    label += f", p={p}"

                draw(ax, data[mask], values[mask], "nx", i, label)
                i += 1

    sizes = np.unique(data["nx"])
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.minorticks_off()
    if np.any(data["backend"] == "cuda"):
        log_y(ax)

    setup(ax, "Grid size nx = ny", "Throughput (Mcells/s)", "Throughput versus grid size")
    legend(ax)

    save(fig, args, "throughput")


def plot_speedup_by_size(args, data, values):
    """Plot the speedup over serial against grid size, one curve per backend and variant.

    Each OpenMP variant is drawn at its largest thread count. The y axis is logarithmic when a
    GPU is in the run, whose speedup is orders of magnitude above the CPU's.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    i = 0
    for backend in np.unique(data["backend"]):
        if backend == "serial":
            continue

        selected = (data["backend"] == backend) & ~np.isnan(values)
        for variant in np.unique(data["variant"][selected]):
            mask = selected & (data["variant"] == variant)
            p = np.max(data["threads"][mask])
            mask = mask & (data["threads"] == p)

            label = f"{backend} {variant}"
            if backend == "openmp":
                label += f", p={p}"

            draw(ax, data[mask], values[mask], "nx", i, label)
            i += 1

    sizes = np.unique(data["nx"])
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.minorticks_off()
    if np.any(data["backend"] == "cuda"):
        log_y(ax)
    ax.axhline(1, color="k", linestyle="--", linewidth=1.3, label="serial")

    setup(ax, "Grid size nx = ny", "Speedup S", "Speedup over serial versus grid size")
    legend(ax)

    save(fig, args, "speedup_size")


def plot_by_threads(args, data, values, ylabel, title, filename, ideal):
    """Plot an OpenMP quantity against threads, with one curve per variant and grid size.

    With more than one variant, only the largest grid is drawn, so the figure stays readable.
    """
    selected = (data["backend"] == "openmp") & ~np.isnan(values)
    variants = np.unique(data["variant"][selected])
    if len(variants) > 1:
        largest = np.max(data["nx"][selected])
        selected = selected & (data["nx"] == largest)
        title += f", nx={largest}"

    omp = data[selected]
    omp_values = values[selected]

    threads = np.unique(omp["threads"])

    fig, ax = plt.subplots(figsize=(8, 5))

    i = 0
    for variant in variants:
        for nx in np.unique(omp["nx"][omp["variant"] == variant]):
            mask = (omp["variant"] == variant) & (omp["nx"] == nx)

            label = f"nx={nx}" if len(variants) == 1 else variant
            draw(ax, omp[mask], omp_values[mask], "threads", i, label)
            i += 1

    if ideal == "linear":
        ax.plot(threads, threads, "k--", linewidth=1.3, label="ideal")
    elif ideal == "unit":
        ax.axhline(1, color="k", linestyle="--", linewidth=1.3, label="ideal")

    ax.set_xticks(threads)
    setup(ax, "Number of threads", ylabel, title)
    ax.legend()

    save(fig, args, filename)


def weak_efficiency(data):
    """Return the weak scaling efficiency of every sample of a weak run.

    The efficiency is the median time on one thread of the same variant over the sample's time,
    since every thread count carries the same cells per thread; it is NaN without a one-thread
    sample. The variant names carry the run, so each run is paired with its own.
    """
    efficiency = np.full(len(data), np.nan)

    for variant in np.unique(data["variant"]):
        same = data["variant"] == variant
        one = data[same & (data["threads"] == 1)]
        if len(one) > 0:
            efficiency[same] = np.median(one["seconds"]) / data["seconds"][same]

    return efficiency


def plot_weak(args, data, values):
    """Plot the weak scaling efficiency against threads, with one curve per variant."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, variant in enumerate(np.unique(data["variant"])):
        mask = (data["variant"] == variant) & ~np.isnan(values)
        draw(ax, data[mask], values[mask], "threads", i, variant)

    ax.axhline(1, color="k", linestyle="--", linewidth=1.3, label="ideal")

    threads = np.unique(data["threads"])
    ax.set_xticks(threads)
    setup(
        ax,
        "Number of threads",
        "Efficiency E(p)",
        f"Weak scaling efficiency, {data['nx'][0]} x {data['nx'][0]} cells per thread",
    )
    ax.legend()

    save(fig, args, "weak_efficiency")


def largest_threads(data, selected):
    """Return the samples of each variant at its largest thread count, among those selected."""
    mask = np.zeros(len(data), dtype=bool)

    for variant in np.unique(data["variant"][selected]):
        same = selected & (data["variant"] == variant)
        mask |= same & (data["threads"] == np.max(data["threads"][same]))

    return mask


def plot_bandwidth(args, data, infos):
    """Plot the achieved memory bandwidth against grid size, one curve per backend and variant.

    The bandwidth is the modelled bytes of a step over the time of a step. Each variant is drawn
    at its largest thread count, under the peaks the runs measured: the GPU's copy kernel and
    the CPU's STREAM triad, and the GPU's peak from its memory clock and bus.
    """
    values = data["bytes_per_step"] * data["nsteps"] / data["seconds"] / 1e9
    fig, ax = plt.subplots(figsize=(8, 5))

    i = 0
    for backend in np.unique(data["backend"]):
        selected = largest_threads(data, data["backend"] == backend)
        for variant in np.unique(data["variant"][selected]):
            mask = selected & (data["variant"] == variant)
            p = data["threads"][mask][0]

            label = f"{backend} {variant}"
            if backend == "openmp":
                label += f", p={p}"

            draw(ax, data[mask], values[mask], "nx", i, label)
            i += 1

    lines = [
        ("gpu_peak_bandwidth", "GPU peak", ":"),
        ("gpu_copy_bandwidth", "GPU copy", "--"),
        ("cpu_triad_bandwidth", "CPU triad", "-."),
    ]
    for key, label, style in lines:
        value = peak(infos, key)
        if value is not None:
            ax.axhline(
                value,
                color="k",
                linestyle=style,
                linewidth=1.3,
                label=f"{label} {value:g} GB/s",
            )

    sizes = np.unique(data["nx"])
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.minorticks_off()
    log_y(ax)

    setup(ax, "Grid size nx = ny", "Bandwidth (GB/s)", "Achieved memory bandwidth versus grid size")
    legend(ax)

    save(fig, args, "bandwidth")


def plot_roofline(args, data, infos):
    """Plot each backend and variant at its largest grid in each precision on the roofline.

    A point sits at its arithmetic intensity, the modelled flops over the modelled bytes of a
    step, and at the flops it achieved. The roofs are the GPU's peak bandwidth, from its memory
    clock and bus, capped by its peak flops in each precision of the runs, and the CPU's triad;
    the CPU has no compute roof, since nothing on the machine reports its peak.
    """
    performance = data["flops_per_step"] * data["nsteps"] / data["seconds"] / 1e9
    fig, ax = plt.subplots(figsize=(8, 5))

    i = 0
    for backend, precision in np.unique(data[["backend", "precision"]]):
        selected = (data["backend"] == backend) & (data["precision"] == precision)
        selected &= data["nx"] == np.max(data["nx"][selected])
        selected = largest_threads(data, selected)
        for variant in np.unique(data["variant"][selected]):
            mask = selected & (data["variant"] == variant)
            intensity = data["flops_per_step"][mask][0] / data["bytes_per_step"][mask][0]

            label = f"{backend} {variant}, nx={data['nx'][mask][0]}"
            ax.plot(
                intensity,
                np.median(performance[mask]),
                marker=MARKERS[i % len(MARKERS)],
                linestyle="none",
                markersize=8,
                label=label,
            )
            i += 1

    x = np.logspace(-2, 2, 200)
    bandwidth = peak(infos, "gpu_peak_bandwidth")
    if bandwidth is not None:
        for precision in np.unique(data["precision"]):
            flops = peak(infos, f"gpu_peak_flops_{precision}")
            roof = bandwidth * x if flops is None else np.minimum(bandwidth * x, flops)
            style = "k-." if precision == "double" else "k-"
            ax.plot(x, roof, style, linewidth=1.3, label=f"GPU roof, {precision}")

    triad = peak(infos, "cpu_triad_bandwidth")
    if triad is not None:
        ax.plot(x, triad * x, "k--", linewidth=1.3, label="CPU triad roof")

    ax.set_xscale("log")
    ax.set_yscale("log")
    setup(
        ax,
        "Arithmetic intensity (flop/byte)",
        "Performance (GFLOPS)",
        "Roofline at the largest grid size",
    )
    ax.grid(which="major", alpha=0.3)
    legend(ax)

    save(fig, args, "roofline")


def remove(args, name):
    """Remove a figure an earlier run of this script left in the report directory."""
    path = os.path.join(args.output_dir, f"{name}_{args.stamp}.png")
    if os.path.exists(path):
        os.remove(path)


def main():
    """Parse the arguments and draw the figures."""
    parser = argparse.ArgumentParser(
        description="Plot the figures of the benchmark runs DIR into REPORT/images."
    )

    parser.add_argument("report", help="report directory, named after its timestamp")
    parser.add_argument(
        "dirs",
        nargs="+",
        help="benchmark run directories, named after their timestamps",
    )
    parser.add_argument("--dpi", type=int, default=120)

    args = parser.parse_args()

    args.stamp = os.path.basename(os.path.normpath(args.report))
    args.output_dir = os.path.join(args.report, "images")
    os.makedirs(args.output_dir, exist_ok=True)

    runs = []
    weak_runs = []
    infos = []
    for directory in args.dirs:
        run = os.path.basename(os.path.normpath(directory))
        path = os.path.join(directory, f"benchmark_{run}.csv")
        info = load_info(path)
        label = info.get("label", run)
        runs.append((path, label))
        infos.append(info)

        weak = os.path.join(directory, f"benchmark_weak_{run}.csv")
        if os.path.exists(weak):
            weak_runs.append((weak, label))

    data = load(runs)
    throughput, speedup, efficiency = calculate(data)

    plot_by_size(args, data, throughput)

    if "bytes_per_step" in data.dtype.names and not np.all(np.isnan(data["bytes_per_step"])):
        modelled = ~np.isnan(data["bytes_per_step"])
        plot_bandwidth(args, data[modelled], infos)
        plot_roofline(args, data[modelled], infos)
    else:
        remove(args, "bandwidth")
        remove(args, "roofline")
        print("==> No modelled traffic in the runs: no bandwidth or roofline plots")

    if weak_runs:
        weak_data = load(weak_runs)
        plot_weak(args, weak_data, weak_efficiency(weak_data))
    else:
        remove(args, "weak_efficiency")

    if np.all(np.isnan(speedup)):
        remove(args, "speedup_size")
        remove(args, "speedup")
        remove(args, "efficiency")
        print("==> No samples with a serial run of the same size: no speedup plots")
        return

    plot_speedup_by_size(args, data, speedup)

    if np.all(np.isnan(speedup[data["backend"] == "openmp"])):
        remove(args, "speedup")
        remove(args, "efficiency")
        print("==> No OpenMP samples with a serial run of the same size: no thread plots")
        return

    plot_by_threads(
        args,
        data,
        speedup,
        "Speedup S(p)",
        "Speedup versus number of threads",
        "speedup",
        ideal="linear",
    )

    plot_by_threads(
        args,
        data,
        efficiency,
        "Efficiency E(p)",
        "Efficiency versus number of threads",
        "efficiency",
        ideal="unit",
    )


if __name__ == "__main__":
    main()
