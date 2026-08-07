include(FetchContent)

# Catch2 v3: an installed copy if one is found, otherwise the pinned release.
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS 3)
FetchContent_MakeAvailable(Catch2)

include(Catch)

# Physics test helpers, on the public API only; linked by the tests, never installed.
add_library(fdtd_test_support STATIC
    ${PROJECT_SOURCE_DIR}/src/fdtd/test/fdtd_test_support.cpp)
target_link_libraries(fdtd_test_support PUBLIC fdtd::fdtd PRIVATE fdtd_warnings)

add_executable(fdtd_tests
    test_backend.cpp
    test_coeffs2d.cpp
    test_constants.cpp
    test_courant.cpp
    test_driver.cpp
    test_fields2d.cpp
    test_grid2d.cpp
    test_material2d.cpp
    test_output.cpp
    test_physics.cpp
    test_serial_boundary.cpp
    test_serial_update.cpp
    test_sim2d.cpp
    test_smoke.cpp
    test_support.cpp
    test_version.cpp
    test_waveform.cpp)
target_link_libraries(fdtd_tests PRIVATE fdtd::fdtd fdtd_driver fdtd_test_support fdtd_warnings Catch2::Catch2WithMain)
# Private library headers, for testing the backends directly.
target_include_directories(fdtd_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(fdtd_tests PRIVATE
    FDTD_TEST_PROJECT_VERSION="${PROJECT_VERSION}"
    FDTD_TEST_PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    FDTD_TEST_PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    FDTD_TEST_PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH})

# Catch tags such as [unit] and [physics] become CTest labels.
catch_discover_tests(fdtd_tests ADD_TAGS_AS_LABELS)

# The fdtd program end to end: help, a bad option, and a small run.
add_test(NAME "fdtd program help" COMMAND fdtd_cli --help)
add_test(NAME "fdtd program rejects bad options" COMMAND fdtd_cli --nx)
set_tests_properties("fdtd program rejects bad options" PROPERTIES WILL_FAIL TRUE)
add_test(NAME "fdtd program runs"
    COMMAND fdtd_cli --nx 32 --steps 20 --probe 20,16 --snapshot-every 10
            --output ${CMAKE_CURRENT_BINARY_DIR}/fdtd_program_run)
set_tests_properties("fdtd program help" "fdtd program rejects bad options" "fdtd program runs"
    PROPERTIES LABELS "unit;driver")
