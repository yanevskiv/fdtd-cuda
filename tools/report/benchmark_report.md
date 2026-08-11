---
title: Benchmark report
date: $date
papersize: a4
geometry:
  - margin=0.75in
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# Run

The benchmark directory `$source` holds $samples samples of $configurations configurations. Each
sample times `Sim2D::run` alone on a fresh simulation, after a warm-up. The tables and figures give
the median of the samples of each configuration, and the throughput is the number of cell updates
per second over that median.

# Machine

$machine

# Results

$results

The checksum is the sum of $$E_z$$ over the grid after the run. Every sample of a configuration
should give the same checksum, and the backends should agree within the tolerance of the
precision.

# Figures

$figures
