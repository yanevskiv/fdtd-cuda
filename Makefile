# Thin wrapper over CMakePresets.json. Settings live in the presets.

PRESET ?= debug
BENCH_PRESET ?= bench
BENCH_DIR ?= $(lastword $(sort $(wildcard results/benchmarks/*)))
PROFILE_DIR ?= $(lastword $(sort $(wildcard results/profiles/*)))
REPORT_DIR ?= results/reports/$(shell date +%s%3N)

# clang-format 22 or later, for IndentPPDirectives: Leave; `make format-tools` installs it in .venv.
CLANG_FORMAT_VERSION = 22.1.8
CLANG_FORMAT ?= .venv/bin/clang-format

SOURCES = $(shell git ls-files --cached --others --exclude-standard 'include/*.h' 'include/*.cuh' 'src/*.h' 'src/*.cpp' 'src/*.cu' 'src/*.cuh' \
	'tests/*.cpp' 'tests/*.h' 'samples/*.cpp')

.PHONY: all configure build test install workflow bench bench-double profile report
.PHONY: lint lint-changed format format-tools clean

all: build

configure:
	cmake --preset $(PRESET)

build: configure
	cmake --build --preset $(PRESET)

test: build
	ctest --preset $(PRESET)

install: build
	cmake --install build/out/$(PRESET)

workflow:
	cmake --workflow --preset $(PRESET)

bench:
	cmake --preset $(BENCH_PRESET)
	cmake --build --preset $(BENCH_PRESET) --target bench

bench-double:
	$(MAKE) bench BENCH_PRESET=bench-double

profile:
	cmake --preset profile
	cmake --build --preset profile --target profile

report:
	cmake --preset bench -DFDTD_REPORT_DIR=$(REPORT_DIR) -DFDTD_BENCH_DIR="$(BENCH_DIR)" \
		-DFDTD_PROFILE_DIR=$(PROFILE_DIR)
	cmake --build --preset bench --target report

TIDY = clang-tidy -p build/out/$(PRESET) --quiet --header-filter='^.*/include/fdtd/.*' \
	--warnings-as-errors=readability-identifier-naming
JOBS ?= $(shell nproc)
CHANGED = $(shell { git diff --name-only HEAD; git ls-files --others --exclude-standard; } | sort -u | \
	grep -E '^(include|src|tests|samples)/.*\.(h|cpp|cu|cuh)$$')

lint: configure
	@if [ -n "$(SOURCES)" ]; then $(CLANG_FORMAT) --dry-run --Werror $(SOURCES); fi
	@printf '%s\n' $(filter %.cpp,$(SOURCES)) | xargs -r -P $(JOBS) -n 1 $(TIDY)

lint-changed: configure
	@if [ -n "$(wildcard $(CHANGED))" ]; then $(CLANG_FORMAT) --dry-run --Werror $(wildcard $(CHANGED)); fi
	@printf '%s\n' $(filter %.cpp,$(wildcard $(CHANGED))) | xargs -r -P $(JOBS) -n 1 $(TIDY)

format:
	@if [ -n "$(SOURCES)" ]; then $(CLANG_FORMAT) -i $(SOURCES); fi

format-tools:
	python3 -m venv .venv
	.venv/bin/pip install clang-format==$(CLANG_FORMAT_VERSION)

clean:
	rm -rf build/out build/install
