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
    test_backend.cpp
    test_coeffs2d.cpp
    test_constants.cpp
    test_courant.cpp
    test_fields2d.cpp
    test_grid2d.cpp
    test_material2d.cpp
    test_serial_boundary.cpp
    test_serial_update.cpp
    test_sim2d.cpp
    test_smoke.cpp
    test_version.cpp)
target_link_libraries(fdtd_tests PRIVATE fdtd::fdtd fdtd_warnings Catch2::Catch2WithMain)
# Private library headers, for testing the backends directly.
target_include_directories(fdtd_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(fdtd_tests PRIVATE
    FDTD_TEST_PROJECT_VERSION="${PROJECT_VERSION}"
    FDTD_TEST_PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    FDTD_TEST_PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    FDTD_TEST_PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH})

catch_discover_tests(fdtd_tests)
