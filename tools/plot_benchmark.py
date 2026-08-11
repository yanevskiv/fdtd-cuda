#!/usr/bin/env python3
#
# Draw the figures of a benchmark run into a report: throughput, speedup and efficiency.
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

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load(path):
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


def calculate(data):
    """Return the throughput, speedup and efficiency of every sample.

    The speedup is over the serial median of the same grid size, and only OpenMP samples have
    one; the others are NaN.
    """
    throughput = data["nx"] * data["ny"] * data["nsteps"] / data["seconds"] / 1e6
    speedup = np.full(len(data), np.nan)

    for nx in np.unique(data["nx"]):
        serial = data[(data["backend"] == "serial") & (data["nx"] == nx)]
        parallel = (data["backend"] == "openmp") & (data["nx"] == nx)
        if len(serial) > 0:
            speedup[parallel] = np.median(serial["seconds"]) / data["seconds"][parallel]

    efficiency = speedup / data["threads"]
    return throughput, speedup, efficiency


def statistics(data, values, x_column):
    """Return the x values and the median of the values at each."""
    xs = np.unique(data[x_column])
    ys = []

    for x in xs:
        selected = values[data[x_column] == x]
        ys.append(np.median(selected))

    return xs, ys


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


def plot_by_size(args, data, values):
    """Plot throughput against grid size, with one curve per backend and thread count."""
    fig, ax = plt.subplots(figsize=(8, 5))

    i = 0
    for backend in np.unique(data["backend"]):
        selected = data["backend"] == backend
        threads = np.unique(data["threads"][selected])

        for p in threads:
            mask = selected & (data["threads"] == p)
            x, y = statistics(data[mask], values[mask], "nx")

            label = backend if len(threads) == 1 else f"{backend}, p={p}"
            ax.plot(
                x,
                y,
                marker=MARKERS[i % len(MARKERS)],
                linewidth=1.8,
                label=label,
            )
            i += 1

    sizes = np.unique(data["nx"])
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.minorticks_off()

    setup(ax, "Grid size nx = ny", "Throughput (Mcells/s)", "Throughput versus grid size")
    ax.legend()

    save(fig, args, "throughput")


def plot_by_threads(args, data, values, ylabel, title, filename, ideal):
    """Plot an OpenMP quantity against threads, with one curve per grid size."""
    selected = (data["backend"] == "openmp") & ~np.isnan(values)
    omp = data[selected]
    omp_values = values[selected]

    threads = np.unique(omp["threads"])

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, nx in enumerate(np.unique(omp["nx"])):
        mask = omp["nx"] == nx
        x, y = statistics(omp[mask], omp_values[mask], "threads")

        ax.plot(
            x,
            y,
            marker=MARKERS[i % len(MARKERS)],
            linewidth=1.8,
            label=f"nx={nx}",
        )

    if ideal == "linear":
        ax.plot(threads, threads, "k--", linewidth=1.3, label="ideal")
    elif ideal == "unit":
        ax.axhline(1, color="k", linestyle="--", linewidth=1.3, label="ideal")

    ax.set_xticks(threads)
    setup(ax, "Number of threads", ylabel, title)
    ax.legend()

    save(fig, args, filename)


def remove(args, name):
    """Remove a figure an earlier run of this script left in the report directory."""
    path = os.path.join(args.output_dir, f"{name}_{args.stamp}.png")
    if os.path.exists(path):
        os.remove(path)


def main():
    """Parse the arguments and draw the figures."""
    parser = argparse.ArgumentParser(
        description="Plot the figures of a benchmark run DIR into REPORT/images."
    )

    parser.add_argument("dir", help="benchmark run directory, named after its timestamp")
    parser.add_argument("report", help="report directory, named after its timestamp")
    parser.add_argument("--dpi", type=int, default=120)

    args = parser.parse_args()

    run = os.path.basename(os.path.normpath(args.dir))
    args.stamp = os.path.basename(os.path.normpath(args.report))
    args.output_dir = os.path.join(args.report, "images")
    os.makedirs(args.output_dir, exist_ok=True)

    data = load(os.path.join(args.dir, f"benchmark_{run}.csv"))
    throughput, speedup, efficiency = calculate(data)

    plot_by_size(args, data, throughput)

    if np.all(np.isnan(speedup)):
        remove(args, "speedup")
        remove(args, "efficiency")
        print("==> No OpenMP samples with a serial run of the same size: no speedup plots")
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
