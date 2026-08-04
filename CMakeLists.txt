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

# The library.
add_library(fdtd
    src/fdtd/TMz/fdtd_tmz_backend.cpp
    src/fdtd/TMz/fdtd_tmz_fields2d.cpp
    src/fdtd/TMz/fdtd_tmz_grid2d.cpp
    src/fdtd/TMz/serial/fdtd_tmz_serial_backend.cpp)
add_library(fdtd::fdtd ALIAS fdtd)
target_compile_features(fdtd PUBLIC cxx_std_20)
target_include_directories(fdtd PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(fdtd PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(fdtd PUBLIC
    FDTD_DOUBLE=$<BOOL:${FDTD_DOUBLE}>
    FDTD_WITH_OPENMP=$<BOOL:${FDTD_WITH_OPENMP}>
    FDTD_WITH_CUDA=$<BOOL:${FDTD_WITH_CUDA}>)
target_link_libraries(fdtd PRIVATE $<BUILD_INTERFACE:fdtd_warnings>)

if(FDTD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation and package export.
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS fdtd EXPORT fdtdTargets)
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
