$intro $samples samples of $configurations configurations. Each sample times
`Simulation::run` alone on a fresh simulation, after a warm-up. The tables and figures give the
median of the samples of each configuration, and the throughput is the number of cell updates per
second over that median. The spread is the interquartile range: the IQR column of the tables, and
the error bars of the figures, which span the middle half of the samples.

## Results

$results

The checksum is the sum of $$E_z$$ over the grid after the run. Every sample of a configuration
should give the same checksum, and the backends should agree within the tolerance of the
precision.

## Weak scaling

$weak

A weak run keeps the cells per thread fixed: on p threads the grid is nx by p nx.

## Figures

$figures
