# Thin wrapper over CMakePresets.json. Settings live in the presets.

PRESET ?= debug

SOURCES = $(shell git ls-files --cached --others --exclude-standard 'include/*.h' 'src/*.h' 'src/*.cpp' 'src/*.cu' 'src/*.cuh' \
	'tests/*.cpp' 'tests/*.h' 'samples/*.cpp')

.PHONY: all configure build test install workflow lint lint-changed format clean

all: build

configure:
	cmake --preset $(PRESET)

build: configure
	cmake --build --preset $(PRESET)

test: build
	ctest --preset $(PRESET)

install: build
	cmake --install build/$(PRESET)

workflow:
	cmake --workflow --preset $(PRESET)

TIDY = clang-tidy -p build/$(PRESET) --quiet --header-filter='^.*/(include/fdtd|tests/support)/.*' \
	--warnings-as-errors=readability-identifier-naming
JOBS ?= $(shell nproc)
CHANGED = $(shell { git diff --name-only HEAD; git ls-files --others --exclude-standard; } | sort -u | \
	grep -E '^(include|src|tests|samples)/.*\.(h|cpp|cu|cuh)$$')

lint: configure
	@if [ -n "$(SOURCES)" ]; then clang-format --dry-run --Werror $(SOURCES); fi
	@printf '%s\n' $(filter %.cpp,$(SOURCES)) | xargs -r -P $(JOBS) -n 1 $(TIDY)

lint-changed: configure
	@if [ -n "$(wildcard $(CHANGED))" ]; then clang-format --dry-run --Werror $(wildcard $(CHANGED)); fi
	@printf '%s\n' $(filter %.cpp,$(wildcard $(CHANGED))) | xargs -r -P $(JOBS) -n 1 $(TIDY)

format:
	@if [ -n "$(SOURCES)" ]; then clang-format -i $(SOURCES); fi

clean:
	rm -rf build
