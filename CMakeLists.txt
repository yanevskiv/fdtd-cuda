cmake_minimum_required(VERSION 3.25)

project(fdtd
    VERSION 0.1.0
    DESCRIPTION "2D TMz FDTD with serial, OpenMP and CUDA backends"
    HOMEPAGE_URL "https://github.com/yanevskiv/fdtd-cuda"
    LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FdtdWarnings)
include(FdtdBackends)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FDTD_DOUBLE "Use double instead of float for fdtd::FloatType" OFF)
option(FDTD_BUILD_TESTS "Build the Catch2 tests" ${PROJECT_IS_TOP_LEVEL})
option(FDTD_WERROR "Treat compiler warnings as errors" OFF)
option(FDTD_BUILD_SAMPLES "Build the samples" ${PROJECT_IS_TOP_LEVEL})
option(FDTD_EXACT "Build the library without contracted multiply-adds, for bit-for-bit backend tests" OFF)

# The library.
add_library(fdtd
    src/fdtd/fdtd_cuda_error.cpp
    src/fdtd/TMz/fdtd_tmz_backend.cpp
    src/fdtd/TMz/fdtd_tmz_coeffs.cpp
    src/fdtd/TMz/fdtd_tmz_fields.cpp
    src/fdtd/TMz/fdtd_tmz_grid.cpp
    src/fdtd/TMz/fdtd_tmz_material.cpp
    src/fdtd/TMz/fdtd_tmz_output.cpp
    src/fdtd/TMz/fdtd_tmz_simulation.cpp
    src/fdtd/TMz/fdtd_tmz_waveform.cpp
    src/fdtd/TMz/backends/fdtd_tmz_backend_serial.cpp
    src/fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_base.cpp
    src/fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_block.cpp
    src/fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_partition.cpp
    src/fdtd/TMz/backends/cpu/fdtd_tmz_backend_cpu_stepper.cpp)
add_library(fdtd::fdtd ALIAS fdtd)
target_compile_features(fdtd PUBLIC cxx_std_20)
target_include_directories(fdtd PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(fdtd PUBLIC
    FDTD_DOUBLE=$<BOOL:${FDTD_DOUBLE}>
    FDTD_WITH_OPENMP=$<BOOL:${FDTD_WITH_OPENMP}>
    FDTD_WITH_CUDA=$<BOOL:${FDTD_WITH_CUDA}>
    FDTD_EXACT=$<BOOL:${FDTD_EXACT}>)
target_link_libraries(fdtd PRIVATE $<BUILD_INTERFACE:fdtd_warnings>)
if(FDTD_WITH_OPENMP)
    target_sources(fdtd PRIVATE src/fdtd/TMz/backends/fdtd_tmz_backend_omp.cpp)
    target_link_libraries(fdtd PRIVATE OpenMP::OpenMP_CXX)
endif()
if(FDTD_WITH_CUDA)
    target_sources(fdtd PRIVATE src/fdtd/TMz/backends/gpu/fdtd_tmz_backend_gpu_device.cu)
    target_link_libraries(fdtd PRIVATE CUDA::cudart)
endif()

# The exact build: no multiply-add is contracted, so rounding follows the source.
if(FDTD_EXACT)
    target_compile_options(fdtd PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

# Argument parsing shared by the programs.
add_library(fdtd_args STATIC src/fdtd/fdtd_args.cpp)
target_link_libraries(fdtd_args PUBLIC fdtd::fdtd PRIVATE fdtd_warnings)

# The command-line driver: a static library the tests link, and the fdtd program.
add_library(fdtd_driver STATIC src/fdtd/fdtd_driver.cpp)
target_link_libraries(fdtd_driver PUBLIC fdtd::fdtd PRIVATE fdtd_args fdtd_warnings)

add_executable(fdtd_cli src/main.cpp)
set_target_properties(fdtd_cli PROPERTIES OUTPUT_NAME fdtd)
target_link_libraries(fdtd_cli PRIVATE fdtd_driver fdtd_warnings)

# The benchmark: a static library the tests link, and the benchmark program.
add_library(fdtd_benchmark STATIC src/fdtd/fdtd_benchmark.cpp)
target_link_libraries(fdtd_benchmark PUBLIC fdtd::fdtd PRIVATE fdtd_args fdtd_warnings)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE fdtd_benchmark fdtd_warnings)

# The benchmark sweep of the config file into build/results/, run by hand; timing is never a test.
add_custom_target(bench
    COMMAND ${PROJECT_SOURCE_DIR}/tools/bench_sweep.sh -p $<TARGET_FILE:benchmark>
    DEPENDS benchmark
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build
    USES_TERMINAL
    VERBATIM)

# The profile run of the config file into build/results/, run by hand with the profile preset.
add_custom_target(profile
    COMMAND ${PROJECT_SOURCE_DIR}/tools/profile_run.sh -p $<TARGET_FILE:benchmark>
    DEPENDS benchmark
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build
    USES_TERMINAL
    VERBATIM)

# The report in FDTD_REPORT_DIR of the runs FDTD_BENCH_DIR and FDTD_PROFILE_DIR, either of which
# may be empty: figures, Markdown and, with pandoc, a PDF.
set(FDTD_REPORT_DIR "" CACHE STRING "Output directory of the report target")
set(FDTD_BENCH_DIR "" CACHE STRING "Benchmark run directory of the report target")
set(FDTD_PROFILE_DIR "" CACHE STRING "Profile run directory of the report target")
find_program(FDTD_PANDOC pandoc)
string(REGEX REPLACE "/+$" "" report_dir "${FDTD_REPORT_DIR}")
string(REGEX REPLACE "/+$" "" bench_run "${FDTD_BENCH_DIR}")
string(REGEX REPLACE "/+$" "" profile_run "${FDTD_PROFILE_DIR}")
get_filename_component(report_stamp "${report_dir}" NAME)
set(report_file ${report_dir}/report_${report_stamp})
if(NOT report_dir OR (NOT bench_run AND NOT profile_run))
    set(report_steps
        COMMAND ${CMAKE_COMMAND} -E echo
            "report: FDTD_REPORT_DIR and FDTD_BENCH_DIR or FDTD_PROFILE_DIR must name directories"
        COMMAND ${CMAKE_COMMAND} -E false)
else()
    set(report_steps COMMAND ${CMAKE_COMMAND} -E make_directory ${report_dir})
    set(report_runs)
    if(bench_run)
        list(APPEND report_steps
            COMMAND ${PROJECT_SOURCE_DIR}/tools/plot_benchmark.py ${bench_run} ${report_dir})
        list(APPEND report_runs -b ${bench_run})
    endif()
    if(profile_run)
        list(APPEND report_runs -p ${profile_run})
    endif()
    list(APPEND report_steps
        COMMAND ${PROJECT_SOURCE_DIR}/tools/tabulate_report.py ${report_dir} ${report_runs})
    if(FDTD_PANDOC)
        list(APPEND report_steps
            COMMAND ${FDTD_PANDOC} --from=markdown --pdf-engine=lualatex --columns=1000
                --resource-path=${report_dir} ${report_file}.md -o ${report_file}.pdf
            COMMAND ${CMAKE_COMMAND} -E echo "==> Creating PDF: ${report_file}.pdf")
    else()
        list(APPEND report_steps
            COMMAND ${CMAKE_COMMAND} -E echo "report: pandoc is missing, so no PDF")
    endif()
endif()
add_custom_target(report ${report_steps}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    USES_TERMINAL
    VERBATIM)

if(FDTD_BUILD_SAMPLES)
    add_executable(sample_point_source samples/sample_point_source.cpp)
    target_link_libraries(sample_point_source PRIVATE fdtd::fdtd fdtd_warnings)
    add_executable(sample_pml samples/sample_pml.cpp)
    target_link_libraries(sample_pml PRIVATE fdtd::fdtd fdtd_warnings)
endif()

if(FDTD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation and package export.
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN backends EXCLUDE
    PATTERN fdtd_args.h EXCLUDE
    PATTERN fdtd_benchmark.h EXCLUDE
    PATTERN fdtd_driver.h EXCLUDE)
install(TARGETS fdtd EXPORT fdtdTargets)
install(TARGETS fdtd_cli benchmark RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(PROGRAMS
    tools/bench_sweep.sh
    tools/plot_benchmark.py
    tools/plot_sim.py
    tools/profile_run.sh
    tools/sysinfo.sh
    tools/tabulate_report.py
    DESTINATION ${CMAKE_INSTALL_BINDIR}/tools)
install(FILES
    tools/report/benchmark_section.md
    tools/report/performance_report.md
    tools/report/profile_section.md
    DESTINATION ${CMAKE_INSTALL_BINDIR}/tools/report)
install(EXPORT fdtdTargets
    NAMESPACE fdtd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fdtd)
configure_package_config_file(cmake/fdtdConfig.cmake.in
    ${PROJECT_BINARY_DIR}/fdtdConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fdtd)
write_basic_package_version_file(${PROJECT_BINARY_DIR}/fdtdConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
install(FILES
    ${PROJECT_BINARY_DIR}/fdtdConfig.cmake
    ${PROJECT_BINARY_DIR}/fdtdConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fdtd)

message(STATUS "fdtd ${PROJECT_VERSION}: OpenMP=${FDTD_WITH_OPENMP} CUDA=${FDTD_WITH_CUDA} double=${FDTD_DOUBLE} exact=${FDTD_EXACT} tests=${FDTD_BUILD_TESTS}")
