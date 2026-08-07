# Thin wrapper over CMakePresets.json. Settings live in the presets.

PRESET ?= debug

SOURCES = $(shell git ls-files --cached --others --exclude-standard 'include/*.h' 'src/*.h' 'src/*.cpp' 'src/*.cu' 'src/*.cuh' \
	'tests/*.cpp' 'tests/*.h' 'samples/*.cpp')

.PHONY: all configure build test install workflow lint format clean

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

lint: configure
	@if [ -n "$(SOURCES)" ]; then clang-format --dry-run --Werror $(SOURCES); fi
	@if [ -n "$(filter %.cpp,$(SOURCES))" ]; then \
		clang-tidy -p build/$(PRESET) --quiet --header-filter='^.*/include/fdtd/.*' \
			--warnings-as-errors=readability-identifier-naming $(filter %.cpp,$(SOURCES)); fi

format:
	@if [ -n "$(SOURCES)" ]; then clang-format -i $(SOURCES); fi

clean:
	rm -rf build
