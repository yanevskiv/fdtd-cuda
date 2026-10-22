<div align="center">

![Fdtd](docs/splash/fdtd-cuda.jpg)

**libfdtd-cuda: A 2D electromagnetic wave simulation library using FDTD with serial, OpenMP and CUDA backends!**

[![License](https://img.shields.io/badge/License-GPLv3.0-blue)](LICENSE)

</div>

---

**libfdtd-cuda** (or **fdtd::CUDA**) is a C++20 library for 2D TMz electromagnetic wave simulation on a Yee grid, with a split-field PML and PEC or PMC outer boundaries. The same simulation runs on a serial, an OpenMP or a CUDA backend, picked by one option. It is the sister project of [fdtd-openmp](https://github.com/yanevskiv/fdtd-openmp), built to practice HPC on the GPU.

The serial backend is the reference. The OpenMP and CUDA backends each come in four optimization levels, `v1` to `v4`, every level one optimization on top of the one before. Every level matches the serial backend within its tolerance, and bit for bit in the `exact` build.

Here's a 512 by 512 grid with a PML, a Gaussian pulse and a probe, run on the GPU:

```cpp
#include <fdtd/fdtd.h>

#include <iostream>

int main()
{
    // A 512 x 512 grid of 1 mm cells with a 20-cell PML, run on the GPU.
    fdtd::TMz::Simulation::Options options;
    options.boundary = fdtd::TMz::Boundary::Pml;
    options.backend = fdtd::TMz::BackendType::Cuda;
    fdtd::TMz::Simulation sim(512, 512, options);

    // A Gaussian pulse on Ez at the centre, and a probe 100 cells to its right.
    sim.add_source(256, 256, fdtd::TMz::Component::Ez,
                   fdtd::TMz::Gaussian::from_steps(sim.dt(), 20));
    const std::size_t probe = sim.add_probe(356, 256, fdtd::TMz::Component::Ez);

    // Run 1000 steps and print the last value the probe recorded.
    sim.run(1000);
    std::cout << sim.probe_data(probe).back() << '\n';
}
```

## Getting started

Build and run the tests with the default `debug` preset:

```sh
make test
```

`make` is a thin wrapper over `CMakePresets.json`; `make test PRESET=release` picks another preset. The CUDA backend is built when `nvcc` is found and OpenMP when the compiler supports it; the tests of a backend are built only with it.

Run a simulation with the `fdtd` driver and plot it:

```sh
build/out/debug/fdtd --nx 256 --steps 400 --boundary pml --backend cuda \
    --probe 160,128 --snapshot-every 50 --output out
tools/plot_sim.py out
```

The driver writes each probe to `out/probe_<k>.csv` and Ez to `out/snapshot_<step>.bin`. `plot_sim.py` draws a heatmap per snapshot and a plot per probe into the same directory, and a GIF with `--animate`. `fdtd --help` lists every option.

## Using the library

Install the library, the programs and the tools into `build/install/`:

```sh
make install PRESET=release
```

Then link it from CMake with `find_package`:

```cmake
find_package(fdtd REQUIRED)
target_link_libraries(app PRIVATE fdtd::fdtd)
```

Configure your project with `-DCMAKE_PREFIX_PATH=<fdtd-cuda>/build/install`. The library is built in `float` by default; configure it with `-DFDTD_DOUBLE=ON` for `double`.

## Backends and levels

`Options::backend` (or `--backend`) picks the backend, and `Options::variant` (or `--variant`) its level. The default of each backend is its highest level.

| Backend | Level | What it adds |
| --- | --- | --- |
| Serial | | The reference: plain loops over the H and E updates. |
| OpenMP | `v1` | A static `omp for` over the rows. |
| OpenMP | `v2` | `omp simd` on the inner loops. |
| OpenMP | `v3` | The core cells read the one tuple of coefficients they share instead of six arrays. |
| OpenMP | `v4` | H and E in one sweep of each thread's band of rows per step. |
| CUDA | `v1` | An H kernel and an E kernel, one thread per cell. |
| CUDA | `v2` | The steps replayed from a captured CUDA graph. |
| CUDA | `v3` | The cells outside the PML read a table of the distinct coefficient tuples. |
| CUDA | `v4` | One kernel per step, each block keeping the H of its tile and its halo in shared memory. |

A variant also takes the backend's runtime settings: `v4+close` or `v4+spread` binds the OpenMP threads, and `v4:16x16` sets the CUDA thread block. `v4+async` copies snapshots to the host while the GPU keeps stepping.

Measured on an Intel Core i7-11700F (8 cores, 12 threads for OpenMP) and an NVIDIA GeForce RTX 3060 (12 GiB), on a 4096 by 4096 grid with a PML, in millions of cell updates per second (Mcells/s), the median of 9 runs of 200 steps:

| Backend | Level | float | double |
| --- | --- | ---: | ---: |
| Serial | | 442 | 223 |
| OpenMP | `v2` | 520 | 259 |
| OpenMP | `v4` | 1114 | 562 |
| CUDA | `v1` | 5712 | 2831 |
| CUDA | `v4` | 13447 | 4914 |

In `float`, CUDA `v4` runs 30 times the serial backend and 12 times OpenMP `v4`. It reaches about 89% of the card's memory bandwidth by the traffic model, and holds its speed up to the largest grid that fits, 13754 by 13754.

## Samples

`samples/` holds small programs on the public API, built with the tests:

| Sample | What it shows |
| --- | --- |
| [Point source](samples/sample_point_source.cpp) | A Gaussian pulse in a PEC box, its arrival at a probe against the speed of light. |
| [PML](samples/sample_pml.cpp) | The reflection of a 20-cell PML at normal incidence. |

Each one prints its numbers and writes a snapshot for `plot_sim.py`.

## Benchmarking

`make bench` builds the `bench` preset (Release) and runs `tools/bench_sweep.sh` over the backends, levels, thread counts and grid sizes in `config`. Copy `config.example` to `config` to change them. Results go to `results/benchmarks/<run>/`, one CSV row per sample, with the machine's details as `# key: value` lines at the top.

```sh
cp config.example config
make bench
```

`make bench-double` does the same in `double`. `make profile` runs the tools named in `config` (`perf`, `valgrind`, `nsys`, `ncu`) into `results/profiles/<run>/`, and `make report` draws the latest benchmark and profile into a Markdown report under `results/reports/`, and a PDF when `pandoc` is installed.

`benchmark --help` lists the options for a single run:

```sh
build/out/bench/benchmark --nx 1024 --backend cuda --variant v1 --variant v4 --output bench.csv
```

## Presets

| Preset | Build |
| --- | --- |
| `debug` | Debug, gcc. The default. |
| `release` | Release, gcc. |
| `clang-debug`, `clang-release` | The same with clang. |
| `cuda` | Debug, CUDA required, warnings as errors. |
| `ci-cpu` | Debug, CPU backends only, warnings as errors. |
| `exact` | Release with no contracted multiply-adds, so every backend matches serial bit for bit. |
| `bench`, `bench-double`, `bench-native` | Release for benchmarks, without tests or samples; in `double`; with `-march=native`. |
| `profile` | `bench` with symbols, frame pointers and CUDA line info. |

## Project structure

```
.
├── include/fdtd                        - Public headers
│   ├── TMz                             - 2D TMz simulation: grid, material, fields, sources, output
│   │   └── backends                    - Serial, OpenMP and CUDA backends (not installed)
│   │       ├── gpu                     - CUDA levels and device helpers
│   │       └── omp                     - OpenMP levels and thread team
│   └── fdtd.h                          - Umbrella header
├── src
│   ├── fdtd                            - Library sources, laid out as include/
│   ├── main.cpp                        - fdtd, the simulation driver
│   └── benchmark.cpp                   - benchmark, the performance measurement program
├── samples                             - Sample programs on the public API
├── tests                               - Catch2 unit, backend and physics tests
├── tools                               - Benchmark, profile, plot and report scripts
├── cmake                               - Backend detection, warnings, package config
├── docs/splash                         - Banner
├── CMakeLists.txt                      - Build, install and package export
├── CMakePresets.json                   - Build presets
├── Makefile                            - Wrapper over the presets
└── config.example                      - Benchmark and profile settings
```

## Requirements

- CMake 3.25 or newer, Ninja, and a C++20 compiler (GCC 11 or newer, or Clang).
- For the CUDA backend: the CUDA Toolkit 12 with `nvcc` and an NVIDIA GPU.
- For the OpenMP backend: a compiler with OpenMP, as GCC and Clang with `libomp`.
- Catch2 is fetched by CMake for the tests.

The plot and report scripts use `numpy` and `matplotlib`:

```sh
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib
```

`make lint` and `make format` need clang-format 22, which `make format-tools` installs into `.venv`, and clang-tidy.

## Copyright

Copyright (C) 2026 Ivan Janevski

`fdtd-cuda` is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

`fdtd-cuda` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with `fdtd-cuda`; if not, see <https://www.gnu.org/licenses/>.
