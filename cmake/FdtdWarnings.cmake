# Warning flags for the project's own targets. Not exported.

add_library(fdtd_warnings INTERFACE)

target_compile_options(fdtd_warnings INTERFACE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        -Wold-style-cast -Wnon-virtual-dtor -Woverloaded-virtual
        -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough>
    $<$<AND:$<BOOL:${FDTD_WERROR}>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>>:-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    $<$<AND:$<BOOL:${FDTD_WERROR}>,$<COMPILE_LANGUAGE:CUDA>>:-Werror=all-warnings -Xcompiler=-Werror>)
