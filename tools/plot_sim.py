#!/usr/bin/env python3
#
# Plot the output of the fdtd driver: Ez heatmaps, probe traces and an animation.
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
import glob
import struct
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

MAGIC = b"FDTDSNAP"
HEADER = struct.Struct("<8sIIQQQddd")


def read_snapshot(path):
    """Read a version 1 snapshot file.

    Returns a dict with the step, the time in seconds, the cell sizes dx and dy in meters,
    and Ez as a float64 array of shape (ny, nx).
    """
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: not a snapshot file")

    magic, version, size, nx, ny, step, dt, dx, dy = HEADER.unpack_from(raw)
    if magic != MAGIC or version != 1 or size not in (4, 8):
        raise ValueError(f"{path}: not a version 1 snapshot file")

    dtype = "<f4" if size == 4 else "<f8"
    ez = np.frombuffer(raw, dtype=dtype, offset=HEADER.size)
    if ez.size != nx * ny:
        raise ValueError(f"{path}: length does not match the header")

    return {
        "step": step,
        "time": step * dt,
        "dx": dx,
        "dy": dy,
        "ez": ez.reshape(ny, nx).astype(np.float64),
    }


def read_probe(path):
    """Read a probe CSV file.

    Returns the first comment line, without its '#', and the rows as a structured array with
    the columns of the header (step, time, ez).
    """
    comment = ""
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                rows.append(line)
            elif not comment:
                comment = line[1:].strip()

    data = np.genfromtxt(rows, delimiter=",", names=True)
    return comment, data


def save(args, fig, name):
    """Save a figure as name in the output directory and close it."""
    os.makedirs(args.output_dir, exist_ok=True)

    path = os.path.join(args.output_dir, name)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("==> Creating image:", path)


def color_limit(snaps):
    """Return the largest |Ez| over the snapshots, or 1 when they are all zero."""
    peak = max(np.max(np.abs(snap["ez"])) for snap in snaps)
    return peak if peak > 0 else 1.0


def title(snap):
    """Return the title of a snapshot: its step and its time in picoseconds."""
    return f"$E_z$ at step {snap['step']}, t = {snap['time'] * 1e12:.1f} ps"


def draw_heatmap(ax, snap, limit):
    """Draw Ez of a snapshot on a color scale of -limit to limit and return the image."""
    ny, nx = snap["ez"].shape
    extent = [0, nx * snap["dx"] * 1e3, 0, ny * snap["dy"] * 1e3]

    image = ax.imshow(
        snap["ez"],
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        extent=extent,
        interpolation="nearest",
    )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title(snap))

    return image


def plot_heatmaps(args, snaps):
    """Plot one heatmap per snapshot, each on its own scale unless --shared-scale is set."""
    shared = color_limit(snaps)

    for snap in snaps:
        limit = shared if args.shared_scale else color_limit([snap])

        fig, ax = plt.subplots(figsize=(6, 5))
        image = draw_heatmap(ax, snap, limit)
        fig.colorbar(image, ax=ax, label="$E_z$ (V/m)")

        save(args, fig, f"{args.prefix}ez_{snap['step']:06d}.png")


def plot_probes(args, probes):
    """Plot the traces of every probe on one figure."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for path, (comment, data) in probes:
        label = comment or os.path.basename(path)
        ax.plot(data["time"] * 1e12, data["ez"], label=label)

    ax.set_xlabel("t (ps)")
    ax.set_ylabel("$E_z$ (V/m)")
    ax.set_title("Probe traces")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save(args, fig, f"{args.prefix}probes.png")


def animate(args, snaps):
    """Write an animated GIF of the snapshots on one shared color scale."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = draw_heatmap(ax, snaps[0], color_limit(snaps))
    fig.colorbar(image, ax=ax, label="$E_z$ (V/m)")

    def update(k):
        """Show snapshot k and return the artists that changed."""
        image.set_data(snaps[k]["ez"])
        ax.set_title(title(snaps[k]))
        return (image,)

    movie = animation.FuncAnimation(fig, update, frames=len(snaps), blit=False)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{args.prefix}ez.gif")
    movie.save(path, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)

    print("==> Creating animation:", path)


def main():
    """Parse the arguments and plot what the input directory holds."""
    parser = argparse.ArgumentParser(
        description="Plot the snapshots and probe traces the fdtd driver wrote to a directory."
    )

    parser.add_argument("input", nargs="?", default="out", help="default: out")
    parser.add_argument("-O", "--output-dir", help="default: INPUT")
    parser.add_argument("--prefix", default="", help="prefix of the image names")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--animate", action="store_true", help="also write ez.gif")
    parser.add_argument("--fps", type=int, default=8, help="frames per second of the GIF")
    parser.add_argument("--shared-scale", action="store_true", help="one scale for all heatmaps")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input

    snaps = []
    for path in glob.glob(os.path.join(args.input, "snapshot_*.bin")):
        snaps.append(read_snapshot(path))
    snaps.sort(key=lambda snap: snap["step"])

    probes = []
    for path in sorted(glob.glob(os.path.join(args.input, "probe_*.csv"))):
        probes.append((path, read_probe(path)))

    if not snaps and not probes:
        parser.error(f"no snapshot_*.bin or probe_*.csv in {args.input}")

    if snaps:
        plot_heatmaps(args, snaps)

    if probes:
        plot_probes(args, probes)

    if args.animate and len(snaps) > 1:
        animate(args, snaps)


if __name__ == "__main__":
    main()
