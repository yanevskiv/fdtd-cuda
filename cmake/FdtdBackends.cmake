# Detection of the optional OpenMP and CUDA backends. Neither is required.

find_package(OpenMP COMPONENTS CXX)
option(FDTD_WITH_OPENMP "Build the OpenMP backend" ${OpenMP_CXX_FOUND})
if(FDTD_WITH_OPENMP AND NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "FDTD_WITH_OPENMP is ON but OpenMP was not found")
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(fdtd_cuda_found ON)
else()
    set(fdtd_cuda_found OFF)
endif()
option(FDTD_WITH_CUDA "Build the CUDA backend" ${fdtd_cuda_found})
if(FDTD_WITH_CUDA AND NOT fdtd_cuda_found)
    message(FATAL_ERROR "FDTD_WITH_CUDA is ON but no CUDA compiler was found")
endif()
