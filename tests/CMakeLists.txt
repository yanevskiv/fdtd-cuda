include(FetchContent)

# Catch2 v3: an installed copy if one is found, otherwise the pinned release.
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS 3)
FetchContent_MakeAvailable(Catch2)

include(Catch)

add_executable(fdtd_tests
    test_fdtd_args.cpp
    test_fdtd_benchmark.cpp
    test_fdtd_constants.cpp
    test_fdtd_cuda_error.cpp
    test_fdtd_driver.cpp
    test_fdtd_tmz_backend.cpp
    test_fdtd_tmz_backend_cpu_block.cpp
    test_fdtd_tmz_backend_cpu_partition.cpp
    test_fdtd_tmz_backend_cpu_stepper.cpp
    test_fdtd_tmz_backend_cuda.cpp
    test_fdtd_tmz_backend_gpu_cell.cpp
    test_fdtd_tmz_backend_omp.cpp
    test_fdtd_tmz_backend_serial.cpp
    test_fdtd_tmz_coeffs.cpp
    test_fdtd_tmz_fields.cpp
    test_fdtd_tmz_grid.cpp
    test_fdtd_tmz_material.cpp
    test_fdtd_tmz_output.cpp
    test_fdtd_tmz_simulation.cpp
    test_fdtd_tmz_waveform.cpp
    test_fdtd_version.cpp
    test_physics.cpp)
if(FDTD_WITH_CUDA)
    target_sources(fdtd_tests PRIVATE
        test_fdtd_tmz_backend_gpu_buffer.cpp
        test_fdtd_tmz_backend_gpu_device.cpp
        test_fdtd_tmz_backend_gpu_stream.cpp)
endif()
target_link_libraries(fdtd_tests PRIVATE fdtd::fdtd fdtd_args fdtd_benchmark fdtd_driver fdtd_warnings Catch2::Catch2WithMain)
# The bound of the backend comparisons, in machine epsilons per step.
set(FDTD_TEST_TOLERANCE 4 CACHE STRING "Bound of the backend comparisons in machine epsilons per step")
# CUDA's bound is wider: nvcc fuses multiply-adds that the host rounds twice.
set(FDTD_TEST_CUDA_TOLERANCE 8 CACHE STRING
    "Bound of the CUDA comparisons with serial in machine epsilons per step")
target_compile_definitions(fdtd_tests PRIVATE
    FDTD_TEST_TOLERANCE=${FDTD_TEST_TOLERANCE}
    FDTD_TEST_CUDA_TOLERANCE=${FDTD_TEST_CUDA_TOLERANCE}
    FDTD_TEST_PROJECT_VERSION="${PROJECT_VERSION}"
    FDTD_TEST_PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    FDTD_TEST_PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    FDTD_TEST_PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH})

# Catch tags such as [unit] and [physics] become CTest labels. OpenMP threads sleep at a
# barrier instead of spinning, since parallel tests share the cores.
catch_discover_tests(fdtd_tests ADD_TAGS_AS_LABELS
    PROPERTIES ENVIRONMENT "OMP_WAIT_POLICY=passive")

# The fdtd program end to end: help, a bad option, and a small run.
add_test(NAME "fdtd program help" COMMAND fdtd_cli --help)
add_test(NAME "fdtd program rejects bad options" COMMAND fdtd_cli --nx)
set_tests_properties("fdtd program rejects bad options" PROPERTIES WILL_FAIL TRUE)
add_test(NAME "fdtd program runs"
    COMMAND fdtd_cli --nx 32 --steps 20 --probe 20,16 --snapshot-every 10
            --output ${CMAKE_CURRENT_BINARY_DIR}/fdtd_program_run)
set_tests_properties("fdtd program help" "fdtd program rejects bad options" "fdtd program runs"
    PROPERTIES LABELS "unit;driver")

# The benchmark program end to end: a short run whose output has the CSV header and a row.
set(bench_header
    "backend,variant,precision,threads,nx,ny,nsteps,repeat,seconds,checksum,bytes_per_step,flops_per_step")
add_test(NAME "benchmark program smoke"
    COMMAND benchmark --nx 64 --steps 10 --repeat 2 --warmup 0)
set_tests_properties("benchmark program smoke" PROPERTIES
    LABELS "bench-smoke"
    PASS_REGULAR_EXPRESSION "${bench_header}\nserial,default,[a-z]+,1,64,64,10,")

# The CUDA tests under compute-sanitizer's four tools, run by hand. Failed API calls are left to
# the tests, which make one on purpose and see every other as a CudaError.
if(FDTD_WITH_CUDA)
    find_program(FDTD_COMPUTE_SANITIZER compute-sanitizer HINTS ${CUDAToolkit_BIN_DIR})
    if(FDTD_COMPUTE_SANITIZER)
        set(sanitize_steps)
        foreach(tool memcheck initcheck synccheck racecheck)
            set(tool_options --tool ${tool} --error-exitcode 1 --report-api-errors no)
            if(tool STREQUAL "memcheck")
                list(APPEND tool_options --leak-check full)
            endif()
            list(APPEND sanitize_steps
                COMMAND ${CMAKE_COMMAND} -E echo "==> compute-sanitizer --tool ${tool}"
                COMMAND ${FDTD_COMPUTE_SANITIZER} ${tool_options} $<TARGET_FILE:fdtd_tests> [cuda])
        endforeach()
    else()
        set(sanitize_steps
            COMMAND ${CMAKE_COMMAND} -E echo "cuda-sanitize: compute-sanitizer is missing"
            COMMAND ${CMAKE_COMMAND} -E false)
    endif()
    add_custom_target(cuda-sanitize ${sanitize_steps}
        DEPENDS fdtd_tests
        USES_TERMINAL
        VERBATIM)
endif()
