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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FDTD_DOUBLE "Use double instead of float for fdtd::FloatType" OFF)
option(FDTD_BUILD_TESTS "Build the Catch2 tests" ${PROJECT_IS_TOP_LEVEL})
option(FDTD_WERROR "Treat compiler warnings as errors" OFF)
option(FDTD_BUILD_SAMPLES "Build the samples" ${PROJECT_IS_TOP_LEVEL})

# The library.
add_library(fdtd
    src/fdtd/TMz/fdtd_tmz_waveform.cpp
    src/fdtd/TMz/fdtd_tmz_backend.cpp
    src/fdtd/TMz/fdtd_tmz_coeffs2d.cpp
    src/fdtd/TMz/fdtd_tmz_fields2d.cpp
    src/fdtd/TMz/fdtd_tmz_grid2d.cpp
    src/fdtd/TMz/fdtd_tmz_material2d.cpp
    src/fdtd/TMz/fdtd_tmz_output.cpp
    src/fdtd/TMz/fdtd_tmz_sim2d.cpp
    src/fdtd/TMz/backends/fdtd_tmz_backend_serial.cpp)
add_library(fdtd::fdtd ALIAS fdtd)
target_compile_features(fdtd PUBLIC cxx_std_20)
target_include_directories(fdtd PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(fdtd PUBLIC
    FDTD_DOUBLE=$<BOOL:${FDTD_DOUBLE}>
    FDTD_WITH_OPENMP=$<BOOL:${FDTD_WITH_OPENMP}>
    FDTD_WITH_CUDA=$<BOOL:${FDTD_WITH_CUDA}>)
target_link_libraries(fdtd PRIVATE $<BUILD_INTERFACE:fdtd_warnings>)

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

# The benchmark sweep, run by hand; timing is never a test.
set(FDTD_BENCH_SIZES "128;256;512;1024;2048" CACHE STRING "Grid sizes of the bench target")
set(FDTD_BENCH_STEPS 500 CACHE STRING "Time steps per sample of the bench target")
set(FDTD_BENCH_REPEAT 10 CACHE STRING "Samples per configuration of the bench target")
add_custom_target(bench
    COMMAND ${PROJECT_SOURCE_DIR}/tools/benchmark.sh -p $<TARGET_FILE:benchmark>
        -n ${FDTD_BENCH_STEPS} -r ${FDTD_BENCH_REPEAT} ${FDTD_BENCH_SIZES}
    DEPENDS benchmark
    USES_TERMINAL
    VERBATIM)

# The report of the benchmark directory FDTD_BENCH_DIR: figures, Markdown and, with pandoc, a PDF.
set(FDTD_BENCH_DIR "" CACHE STRING "Benchmark directory of the bench-report target")
find_program(FDTD_PANDOC pandoc)
if(NOT FDTD_BENCH_DIR)
    set(bench_report_steps
        COMMAND ${CMAKE_COMMAND} -E echo "bench-report: FDTD_BENCH_DIR names no directory"
        COMMAND ${CMAKE_COMMAND} -E false)
else()
    set(bench_report_steps
        COMMAND ${PROJECT_SOURCE_DIR}/tools/plot_benchmark.py ${FDTD_BENCH_DIR}
        COMMAND ${PROJECT_SOURCE_DIR}/tools/tabulate_benchmark.py ${FDTD_BENCH_DIR})
    if(FDTD_PANDOC)
        list(APPEND bench_report_steps
            COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/dist/docs
            COMMAND ${FDTD_PANDOC} --from=markdown --pdf-engine=lualatex --columns=1000
                --resource-path=${FDTD_BENCH_DIR}/report
                ${FDTD_BENCH_DIR}/report/benchmark_report.md
                -o ${PROJECT_SOURCE_DIR}/dist/docs/benchmark_report.pdf)
    else()
        list(APPEND bench_report_steps
            COMMAND ${CMAKE_COMMAND} -E echo "bench-report: pandoc is missing, so no PDF")
    endif()
endif()
add_custom_target(bench-report ${bench_report_steps}
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
install(TARGETS fdtd_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
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

message(STATUS "fdtd ${PROJECT_VERSION}: OpenMP=${FDTD_WITH_OPENMP} CUDA=${FDTD_WITH_CUDA} double=${FDTD_DOUBLE} tests=${FDTD_BUILD_TESTS}")
