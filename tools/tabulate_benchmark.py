#!/usr/bin/env python3
#
# Write the Markdown report of a benchmark run: the machine, the medians and the figures.
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

TEMPLATE = os.path.join(os.path.dirname(__file__), "report", "benchmark_report.md")

CONFIG = ["backend", "variant", "precision", "threads", "nx", "ny", "nsteps"]

FIGURES = [
    ("throughput", "Throughput versus grid size"),
    ("speedup", "Speedup versus number of threads"),
    ("efficiency", "Efficiency versus number of threads"),
]


def load(path):
    """Read a benchmark CSV.

    Returns the '# key: value' lines as (key, value) pairs, and the samples.
    """
    sysinfo = []
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                sysinfo.append((key.strip(), value.strip()))
            else:
                rows.append(line)

    data = np.genfromtxt(
        rows,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    return sysinfo, np.atleast_1d(data)


def run_date(stamp):
    """Return the start of a run in UTC, from its timestamp in milliseconds.

    Returns it as 'January 19, 2026 (14:05:09Z)', the time with its zone as ISO 8601 writes it.
    """
    start = datetime.datetime.fromtimestamp(int(stamp) / 1000, tz=datetime.timezone.utc)
    time = start.isoformat(timespec="seconds").split("T")[1].replace("+00:00", "Z")
    return f"{start:%B} {start.day}, {start:%Y} ({time})"


def machine_table(sysinfo):
    """Return the machine description as a Markdown table."""
    lines = [
        "| Key | Value |",
        "|:--|:--|",
    ]

    for key, value in sysinfo:
        lines.append(f"| `{key}` | {value} |")

    return "\n".join(lines)


def results_table(data):
    """Return the median time, throughput and checksum of each configuration as a table."""
    lines = [
        "| Backend | Variant | Precision | Threads | Grid | Steps | Samples "
        "| Median (s) | Mcells/s | Checksum |",
        "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]

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

        lines.append(
            f"| {backend} | {variant} | {precision} | {threads} | {nx} x {ny} | {nsteps} "
            f"| {len(samples)} | {median:.6f} | {throughput:.1f} | {checksum} |"
        )

    return "\n".join(lines)


def figure_list(directory, stamp):
    """Return the Markdown of the figures plot_benchmark.py drew into DIR/images."""
    lines = []

    for name, caption in FIGURES:
        image = f"images/{name}_{stamp}.png"
        if os.path.exists(os.path.join(directory, image)):
            lines.append(f"![{caption}]({image}){{width=90%}}")

    if not lines:
        return "No figures: run plot_benchmark.py on this directory first."

    if len(lines) < len(FIGURES):
        lines.append(
            "The speedup and efficiency figures are left out: "
            "no OpenMP run has a serial run of the same size."
        )

    return "\n\n".join(lines)


def main():
    """Parse the arguments and write the report."""
    parser = argparse.ArgumentParser(
        description="Write the Markdown report of a benchmark run into DIR."
    )

    parser.add_argument("dir", help="run directory, named after its timestamp")

    args = parser.parse_args()

    stamp = os.path.basename(os.path.normpath(args.dir))
    sysinfo, data = load(os.path.join(args.dir, f"benchmark_{stamp}.csv"))

    with open(TEMPLATE, encoding="utf-8") as f:
        template = string.Template(f.read())

    text = template.substitute(
        date=run_date(stamp),
        stamp=stamp,
        samples=len(data),
        configurations=len(np.unique(data[CONFIG])),
        machine=machine_table(sysinfo),
        results=results_table(data),
        figures=figure_list(args.dir, stamp),
    )

    path = os.path.join(args.dir, f"benchmark_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print("==> Creating report:", path)


if __name__ == "__main__":
    main()
