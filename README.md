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

The tracked source tree is:

```
.
├── cmake                                                  - CMake module directory
│   ├── FdtdBackends.cmake                                 - OpenMP and CUDA detection
│   ├── FdtdWarnings.cmake                                 - Compiler warning flags
│   └── fdtdConfig.cmake.in                                - Package config template
├── docs                                                   - Documentation directory
│   └── splash                                             - Banner directory
│       └── fdtd-cuda.jpg                                  - Project banner
├── include                                                - Header directory
│   └── fdtd                                               - FDTD header directory
│       ├── TMz                                            - 2D TMz simulation header directory
│       │   ├── backends                                   - Backend header directory (not installed)
│       │   │   ├── gpu                                    - CUDA helper header directory
│       │   │   │   ├── fdtd_tmz_backend_gpu_buffer.h      - Device buffer header
│       │   │   │   ├── fdtd_tmz_backend_gpu_cell.h        - One-cell update header
│       │   │   │   ├── fdtd_tmz_backend_gpu_check.cuh     - CUDA call check header
│       │   │   │   ├── fdtd_tmz_backend_gpu_device.h      - Device query header
│       │   │   │   ├── fdtd_tmz_backend_gpu_error.h       - CUDA error header
│       │   │   │   ├── fdtd_tmz_backend_gpu_event.h       - CUDA event header
│       │   │   │   ├── fdtd_tmz_backend_gpu_graph.h       - CUDA graph header
│       │   │   │   ├── fdtd_tmz_backend_gpu_pinned.h      - Page-locked host memory header
│       │   │   │   ├── fdtd_tmz_backend_gpu_points.h      - Source and probe kernel header
│       │   │   │   ├── fdtd_tmz_backend_gpu_stream.h      - CUDA stream header
│       │   │   │   ├── fdtd_tmz_backend_gpu_table.h       - Coefficient table header
│       │   │   │   └── fdtd_tmz_backend_gpu_thread.cuh    - Thread cell and step header
│       │   │   ├── omp                                    - OpenMP helper header directory
│       │   │   │   ├── fdtd_tmz_backend_omp_levels.h      - OpenMP level header
│       │   │   │   ├── fdtd_tmz_backend_omp_team.h        - OpenMP thread team header
│       │   │   │   ├── fdtd_tmz_backend_omp_v2.h          - OpenMP v2 kernel header
│       │   │   │   └── fdtd_tmz_backend_omp_v3.h          - OpenMP v3 kernel header
│       │   │   ├── fdtd_tmz_backend_cuda.h                - CUDA backend header
│       │   │   ├── fdtd_tmz_backend_cuda_options.h        - CUDA settings and variant header
│       │   │   ├── fdtd_tmz_backend_omp.h                 - OpenMP backend header
│       │   │   ├── fdtd_tmz_backend_omp_options.h         - OpenMP settings and variant header
│       │   │   └── fdtd_tmz_backend_serial.h              - Serial backend header
│       │   ├── fdtd_tmz_backend.h                         - Backend interface and registry header
│       │   ├── fdtd_tmz_boundary.h                        - Outer boundary header
│       │   ├── fdtd_tmz_coeffs.h                          - Coefficient header
│       │   ├── fdtd_tmz_component.h                       - Field component header
│       │   ├── fdtd_tmz_fields.h                          - Field header
│       │   ├── fdtd_tmz_grid.h                            - Grid header
│       │   ├── fdtd_tmz_material.h                        - Material header
│       │   ├── fdtd_tmz_output.h                          - Probe CSV and snapshot header
│       │   ├── fdtd_tmz_simulation.h                      - Simulation header
│       │   └── fdtd_tmz_waveform.h                        - Source waveform header
│       ├── fdtd.h                                         - Public API header
│       ├── fdtd_aligned_buffer.h                          - Aligned buffer header
│       ├── fdtd_args.h                                    - Command-line parsing header (not installed)
│       ├── fdtd_benchmark.h                               - Benchmark program header (not installed)
│       ├── fdtd_constants.h                               - Physical constants header
│       ├── fdtd_driver.h                                  - Driver program header (not installed)
│       ├── fdtd_float.h                                   - Floating-point type header
│       └── fdtd_version.h                                 - Version header
├── samples                                                - Sample program directory
│   ├── sample_pml.cpp                                     - PML reflection sample
│   └── sample_point_source.cpp                            - Point source sample
├── src                                                    - Source file directory
│   ├── fdtd                                               - FDTD source file directory
│   │   ├── TMz                                            - 2D TMz simulation source directory
│   │   │   ├── backends                                   - Backend source directory
│   │   │   │   ├── gpu                                    - CUDA source directory
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_buffer.cu     - Device buffer source file
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_device.cu     - Device query source file
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_error.cpp     - CUDA error source file
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_event.cu      - CUDA event source file
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_graph.cu      - CUDA graph source file
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_pinned.cu     - Page-locked host memory source file
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_points.cu     - Source and probe kernels
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_stream.cu     - CUDA stream source file
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_table.cpp     - Coefficient table source file
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_v1.cu         - CUDA level v1, two kernels
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_v2.cu         - CUDA level v2, graph
│   │   │   │   │   ├── fdtd_tmz_backend_gpu_v3.cu         - CUDA level v3, table
│   │   │   │   │   └── fdtd_tmz_backend_gpu_v4.cu         - CUDA level v4, tiled
│   │   │   │   ├── omp                                    - OpenMP source directory
│   │   │   │   │   ├── fdtd_tmz_backend_omp_v1.cpp        - OpenMP level v1, rows
│   │   │   │   │   ├── fdtd_tmz_backend_omp_v2.cpp        - OpenMP level v2, simd
│   │   │   │   │   ├── fdtd_tmz_backend_omp_v3.cpp        - OpenMP level v3, table
│   │   │   │   │   └── fdtd_tmz_backend_omp_v4.cpp        - OpenMP level v4, fused
│   │   │   │   ├── fdtd_tmz_backend_cuda.cu               - CUDA backend source file
│   │   │   │   ├── fdtd_tmz_backend_cuda_options.cpp      - CUDA settings and variant source file
│   │   │   │   ├── fdtd_tmz_backend_omp.cpp               - OpenMP backend source file
│   │   │   │   ├── fdtd_tmz_backend_omp_options.cpp       - OpenMP settings and variant source file
│   │   │   │   └── fdtd_tmz_backend_serial.cpp            - Serial backend source file
│   │   │   ├── fdtd_tmz_backend.cpp                       - Backend interface and registry source file
│   │   │   ├── fdtd_tmz_coeffs.cpp                        - Coefficient source file
│   │   │   ├── fdtd_tmz_fields.cpp                        - Field source file
│   │   │   ├── fdtd_tmz_grid.cpp                          - Grid source file
│   │   │   ├── fdtd_tmz_material.cpp                      - Material source file
│   │   │   ├── fdtd_tmz_output.cpp                        - Probe CSV and snapshot source file
│   │   │   ├── fdtd_tmz_simulation.cpp                    - Simulation source file
│   │   │   └── fdtd_tmz_waveform.cpp                      - Source waveform source file
│   │   ├── fdtd_args.cpp                                  - Command-line parsing source file
│   │   ├── fdtd_benchmark.cpp                             - Benchmark program source file
│   │   └── fdtd_driver.cpp                                - Driver program source file
│   ├── benchmark.cpp                                      - Performance measurement program
│   └── main.cpp                                           - Simulation program
├── tests                                                  - Test directory
│   ├── CMakeLists.txt                                     - Test build rules
│   ├── test_fdtd_args.cpp                                 - Command-line parsing tests
│   ├── test_fdtd_benchmark.cpp                            - Benchmark program tests
│   ├── test_fdtd_constants.cpp                            - Physical constants tests
│   ├── test_fdtd_driver.cpp                               - Driver program tests
│   ├── test_fdtd_tmz_backend.cpp                          - Backend registry tests
│   ├── test_fdtd_tmz_backend_cuda.cpp                     - CUDA backend against serial tests
│   ├── test_fdtd_tmz_backend_cuda_options.cpp             - CUDA settings and variant tests
│   ├── test_fdtd_tmz_backend_gpu_buffer.cpp               - Device buffer tests
│   ├── test_fdtd_tmz_backend_gpu_cell.cpp                 - One-cell update tests
│   ├── test_fdtd_tmz_backend_gpu_device.cpp               - Device query tests
│   ├── test_fdtd_tmz_backend_gpu_error.cpp                - CUDA error tests
│   ├── test_fdtd_tmz_backend_gpu_stream.cpp               - CUDA stream tests
│   ├── test_fdtd_tmz_backend_gpu_table.cpp                - Coefficient table tests
│   ├── test_fdtd_tmz_backend_omp.cpp                      - OpenMP backend against serial tests
│   ├── test_fdtd_tmz_backend_omp_options.cpp              - OpenMP settings and variant tests
│   ├── test_fdtd_tmz_backend_serial.cpp                   - Serial backend tests
│   ├── test_fdtd_tmz_coeffs.cpp                           - Coefficient tests
│   ├── test_fdtd_tmz_fields.cpp                           - Field tests
│   ├── test_fdtd_tmz_grid.cpp                             - Grid tests
│   ├── test_fdtd_tmz_material.cpp                         - Material tests
│   ├── test_fdtd_tmz_output.cpp                           - Probe CSV and snapshot tests
│   ├── test_fdtd_tmz_simulation.cpp                       - Simulation tests
│   ├── test_fdtd_tmz_waveform.cpp                         - Source waveform tests
│   ├── test_fdtd_version.cpp                              - Version and floating-point type tests
│   └── test_physics.cpp                                   - Physics tests
├── tools                                                  - Utility script directory
│   ├── report                                             - Report template directory
│   │   ├── benchmark_section.md                           - Benchmark section template
│   │   ├── performance_report.md                          - Report template
│   │   └── profile_section.md                             - Profile section template
│   ├── bench_sweep.sh                                     - Benchmark sweep script
│   ├── plot_benchmark.py                                  - Performance plotting script
│   ├── plot_sim.py                                        - Simulation plotting script
│   ├── profile_run.sh                                     - Profile run script
│   ├── sysinfo.sh                                         - System information script
│   └── tabulate_report.py                                 - Report table script
├── CMakeLists.txt                                         - Build, install and package rules
├── CMakePresets.json                                      - Build presets
├── LICENSE                                                - GPLv3 license text
├── Makefile                                               - Wrapper over the presets
├── README.md                                              - Project description
└── config.example                                         - Benchmark and profile settings
```

`make install PRESET=release` installs the library, the headers, the programs and the tools into `build/install/`:

```
build/install
├── bin                                      - Executable directory
│   ├── tools                                - Utility script directory
│   │   ├── report                           - Report template directory
│   │   │   ├── benchmark_section.md         - Benchmark section template
│   │   │   ├── performance_report.md        - Report template
│   │   │   └── profile_section.md           - Profile section template
│   │   ├── bench_sweep.sh                   - Benchmark sweep script
│   │   ├── plot_benchmark.py                - Performance plotting script
│   │   ├── plot_sim.py                      - Simulation plotting script
│   │   ├── profile_run.sh                   - Profile run script
│   │   ├── sysinfo.sh                       - System information script
│   │   └── tabulate_report.py               - Report table script
│   ├── benchmark                            - Performance measurement executable
│   └── fdtd                                 - Simulation executable
├── include                                  - Header directory
│   └── fdtd                                 - FDTD header directory
│       ├── TMz                              - 2D TMz simulation header directory
│       │   ├── fdtd_tmz_backend.h           - Backend interface and registry header
│       │   ├── fdtd_tmz_boundary.h          - Outer boundary header
│       │   ├── fdtd_tmz_coeffs.h            - Coefficient header
│       │   ├── fdtd_tmz_component.h         - Field component header
│       │   ├── fdtd_tmz_fields.h            - Field header
│       │   ├── fdtd_tmz_grid.h              - Grid header
│       │   ├── fdtd_tmz_material.h          - Material header
│       │   ├── fdtd_tmz_output.h            - Probe CSV and snapshot header
│       │   ├── fdtd_tmz_simulation.h        - Simulation header
│       │   └── fdtd_tmz_waveform.h          - Source waveform header
│       ├── fdtd.h                           - Public API header
│       ├── fdtd_aligned_buffer.h            - Aligned buffer header
│       ├── fdtd_constants.h                 - Physical constants header
│       ├── fdtd_float.h                     - Floating-point type header
│       └── fdtd_version.h                   - Version header
└── lib                                      - Library directory
    ├── cmake                                - CMake package directory
    │   └── fdtd                             - fdtd package directory
    │       ├── fdtdConfig.cmake             - Package config
    │       ├── fdtdConfigVersion.cmake      - Package version
    │       ├── fdtdTargets-release.cmake    - Exported targets for Release
    │       └── fdtdTargets.cmake            - Exported targets
    └── libfdtd.a                            - Simulation library
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
