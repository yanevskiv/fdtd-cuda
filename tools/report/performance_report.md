---
title: Performance report
subtitle: fdtd-cuda
date: $date
papersize: a4
geometry:
  - margin=0.75in
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{3pt}}
---

# Sources

The report `$stamp` is made from the runs below, each named after its start in milliseconds since
1970. The input directories contain the raw CSV samples and their machine metadata.

$sources

# Machine

The description each run recorded of the machine and the build it ran, the values all runs share first.

$machine

# Benchmark

$benchmark

# Profile

$profile
