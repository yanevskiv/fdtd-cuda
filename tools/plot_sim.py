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

import argparse
import glob
import os
import struct

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

MAGIC = b"FDTDSNAP"
HEADER = struct.Struct("<8sIIQQQddd")


def read_snapshot(path):
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
    return {"step": step, "time": step * dt, "dx": dx, "dy": dy,
            "ez": ez.reshape(ny, nx).astype(np.float64)}


def read_probe(path):
    comment = ""
    with open(path, encoding="utf-8") as f:
        rows = []
        for line in f:
            if line.startswith("#"):
                comment = comment or line[1:].strip()
            else:
                rows.append(line)
    return comment, np.genfromtxt(rows, delimiter=",", names=True)


def save(args, fig, name):
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, name)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("==> Creating image:", path)


def extent_mm(snap):
    ny, nx = snap["ez"].shape
    return [0, nx * snap["dx"] * 1e3, 0, ny * snap["dy"] * 1e3]


def color_limit(snaps):
    peak = max(float(np.max(np.abs(s["ez"]))) for s in snaps)
    return peak if peak > 0 else 1.0


def draw_heatmap(ax, snap, limit):
    image = ax.imshow(snap["ez"], origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                      extent=extent_mm(snap), interpolation="nearest")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"$E_z$ at step {snap['step']}, t = {snap['time'] * 1e12:.1f} ps")
    return image


def plot_heatmaps(args, snaps):
    limit = color_limit(snaps) if args.shared_scale else None
    for snap in snaps:
        fig, ax = plt.subplots(figsize=(6, 5))
        own = limit or color_limit([snap])
        image = draw_heatmap(ax, snap, own)
        fig.colorbar(image, ax=ax, label="$E_z$ (V/m)")
        save(args, fig, f"{args.prefix}ez_{snap['step']:06d}.png")


def plot_probes(args, probes):
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
    limit = color_limit(snaps)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = draw_heatmap(ax, snaps[0], limit)
    fig.colorbar(image, ax=ax, label="$E_z$ (V/m)")

    def update(k):
        snap = snaps[k]
        image.set_data(snap["ez"])
        ax.set_title(f"$E_z$ at step {snap['step']}, t = {snap['time'] * 1e12:.1f} ps")
        return (image,)

    movie = animation.FuncAnimation(fig, update, frames=len(snaps), blit=False)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{args.prefix}ez.gif")
    movie.save(path, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    print("==> Creating animation:", path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the snapshots and probe traces the fdtd driver wrote to a directory."
    )
    parser.add_argument("input", nargs="?", default="out",
                        help="directory with snapshot_*.bin and probe_*.csv (default: out)")
    parser.add_argument("-O", "--output-dir", help="directory for the images (default: INPUT)")
    parser.add_argument("--prefix", default="", help="prefix of the image names")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--animate", action="store_true", help="also write ez.gif")
    parser.add_argument("--fps", type=int, default=8, help="frames per second of the animation")
    parser.add_argument("--shared-scale", action="store_true",
                        help="one color scale for every heatmap instead of one each")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input

    snaps = [read_snapshot(p) for p in sorted(glob.glob(os.path.join(args.input, "snapshot_*.bin")))]
    snaps.sort(key=lambda s: s["step"])
    probes = [(p, read_probe(p)) for p in sorted(glob.glob(os.path.join(args.input, "probe_*.csv")))]
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
