The run `$stamp` profiles the benchmark program, built with the `profile` preset (the Release
flags plus `-g -fno-omit-frame-pointer`), on the $backend backend (threads: $threads), one sample
and no warm-up: a $perf_size x $perf_size grid for $perf_steps steps under perf, and a
$valgrind_size x $valgrind_size grid for $valgrind_steps steps under valgrind. Each tool sees the
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
