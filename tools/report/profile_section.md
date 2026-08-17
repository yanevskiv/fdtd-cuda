The run `$stamp` profiles the benchmark program, built with the `profile` preset (the Release
flags plus `-g -fno-omit-frame-pointer`, and `-lineinfo` for CUDA), on the $backend backend
(threads: $threads), one sample and no warm-up: a $perf_size x $perf_size grid for $perf_steps
steps under perf, nsys and ncu, and a $valgrind_size x $valgrind_size grid for $valgrind_steps
steps under valgrind. Each tool sees the
whole program, setup included, and slows it, so these runs give no benchmark numbers. The raw
output of every tool is kept in the run directory.

## perf stat

Hardware counters over the whole run, in user space only.

$perf_stat

## perf record

The functions with the most samples of user-space cycles.

$perf_record

## callgrind

The functions that execute the most instructions themselves.

$callgrind

## cachegrind

The miss rates of the level-1 instruction (I1) and data (D1) caches and the last level (LL), as
cachegrind simulates them.

$cachegrind

## memcheck

$memcheck

## nsys

The GPU kernels by their total time on the device, as nsys traced them. These times are the
kernels alone; the benchmark's times hold the whole run, copies and launches included.

$nsys

## ncu

The speed of light of $ncu_count kernel launches after the first $ncu_skip, as ncu measured them:
the throughput of memory and of the SMs as a share of the device's peak.

$ncu
