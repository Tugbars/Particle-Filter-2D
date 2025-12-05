cmake_minimum_required(VERSION 3.16)
project(particle_filter_2d C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# MKL configuration
set(MKL_INTERFACE lp64)
set(MKL_THREADING intel_thread)  # Intel OpenMP for MSVC
set(MKL_MPI intelmpi)

# Hint for MKL location on Windows
# Set MKL_DIR or CMAKE_PREFIX_PATH to your oneAPI installation
# Example: cmake -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/oneAPI/mkl/latest" ..
if(WIN32 AND NOT DEFINED ENV{MKLROOT})
    # Common installation paths
    list(APPEND CMAKE_PREFIX_PATH 
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
        "C:/Program Files/Intel/oneAPI/mkl/latest"
    )
endif()

# Find MKL
find_package(MKL CONFIG REQUIRED)

# OpenMP - MKL brings its own Intel OpenMP on Windows
if(MSVC)
    # Intel OpenMP comes with MKL
    set(OpenMP_FOUND TRUE)
else()
    find_package(OpenMP REQUIRED)
endif()

#───────────────────────────────────────────────────────────────────────────────
# Options
#───────────────────────────────────────────────────────────────────────────────

option(BUILD_TESTS "Build test executables" ON)
option(BUILD_1D "Build 1D particle filter" ON)
option(BUILD_2D "Build 2D particle filter (stochastic vol)" ON)
option(PF2D_USE_FLOAT "Use single precision for 2D filter" OFF)
option(PF2D_HIGH_N "Enable HIGH_N mode for 50k+ particles" OFF)

#───────────────────────────────────────────────────────────────────────────────
# Compiler flags
#───────────────────────────────────────────────────────────────────────────────

if(MSVC)
    set(COMMON_FLAGS 
        /W4 
        /openmp:llvm     # Use LLVM OpenMP for modern features (reduction max, C99 loops)
        $<$<CONFIG:Release>:/O2 /fp:fast /arch:AVX2>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
    set(COMMON_DEFS _CRT_SECURE_NO_WARNINGS)
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
    set(COMMON_FLAGS -Wall -xHost -qopenmp -qopt-report=3 -qopt-report-phase=vec)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(COMMON_FLAGS -Wall -Wextra -Wpedantic -O3 -march=native -fopenmp -ffast-math -funroll-loops)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(COMMON_FLAGS -Wall -Wextra -Wpedantic -O3 -march=native -fopenmp -ffast-math -funroll-loops)
endif()

#───────────────────────────────────────────────────────────────────────────────
# 1D Particle Filter
#───────────────────────────────────────────────────────────────────────────────

if(BUILD_1D)
    add_library(particle_filter STATIC
        1D/particle_filter.c
    )

    target_include_directories(particle_filter PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/1D
    )

    target_link_libraries(particle_filter PUBLIC
        MKL::MKL
        $<$<NOT:$<BOOL:${WIN32}>>:OpenMP::OpenMP_C>
    )

    target_compile_options(particle_filter PRIVATE ${COMMON_FLAGS})
    if(DEFINED COMMON_DEFS)
        target_compile_definitions(particle_filter PRIVATE ${COMMON_DEFS})
    endif()

    # Shared library
    add_library(particle_filter_shared SHARED
        1D/particle_filter.c
    )

    target_include_directories(particle_filter_shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/1D
    )

    target_link_libraries(particle_filter_shared PUBLIC
        MKL::MKL
        $<$<NOT:$<BOOL:${WIN32}>>:OpenMP::OpenMP_C>
    )

    target_compile_options(particle_filter_shared PRIVATE ${COMMON_FLAGS})
    if(DEFINED COMMON_DEFS)
        target_compile_definitions(particle_filter_shared PRIVATE ${COMMON_DEFS})
    endif()

    set_target_properties(particle_filter_shared PROPERTIES
        OUTPUT_NAME "particle_filter"
    )

    if(WIN32)
        # Windows: export all symbols or use .def
        set_target_properties(particle_filter_shared PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
        )
    else()
        set_target_properties(particle_filter_shared PROPERTIES PREFIX "")
    endif()
endif()

#───────────────────────────────────────────────────────────────────────────────
# 2D Particle Filter (Stochastic Volatility)
#───────────────────────────────────────────────────────────────────────────────

if(BUILD_2D)
    add_library(particle_filter_2d STATIC
        2D/particle_filter_2d.c
        2D/pf2d_adaptive.c
    )

    target_include_directories(particle_filter_2d PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/2D
    )

    target_link_libraries(particle_filter_2d PUBLIC
        MKL::MKL
        $<$<NOT:$<BOOL:${WIN32}>>:OpenMP::OpenMP_C>
    )

    target_compile_options(particle_filter_2d PRIVATE ${COMMON_FLAGS})
    if(DEFINED COMMON_DEFS)
        target_compile_definitions(particle_filter_2d PRIVATE ${COMMON_DEFS})
    endif()

    # Optional: single precision
    if(PF2D_USE_FLOAT)
        target_compile_definitions(particle_filter_2d PUBLIC PF2D_USE_FLOAT)
    endif()

    # Optional: HIGH_N mode for large particle counts
    if(PF2D_HIGH_N)
        target_compile_definitions(particle_filter_2d PUBLIC PF2D_HIGH_N)
    endif()

    # Shared library
    add_library(particle_filter_2d_shared SHARED
        2D/particle_filter_2d.c
        2D/pf2d_adaptive.c   
    )

    target_include_directories(particle_filter_2d_shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/2D
    )

    target_link_libraries(particle_filter_2d_shared PUBLIC
        MKL::MKL
        $<$<NOT:$<BOOL:${WIN32}>>:OpenMP::OpenMP_C>
    )

    target_compile_options(particle_filter_2d_shared PRIVATE ${COMMON_FLAGS})
    if(DEFINED COMMON_DEFS)
        target_compile_definitions(particle_filter_2d_shared PRIVATE ${COMMON_DEFS})
    endif()

    if(PF2D_USE_FLOAT)
        target_compile_definitions(particle_filter_2d_shared PUBLIC PF2D_USE_FLOAT)
    endif()

    if(PF2D_HIGH_N)
        target_compile_definitions(particle_filter_2d_shared PUBLIC PF2D_HIGH_N)
    endif()

    set_target_properties(particle_filter_2d_shared PROPERTIES
        OUTPUT_NAME "particle_filter_2d"
    )

    if(WIN32)
        # Use explicit .def file for controlled exports
        target_sources(particle_filter_2d_shared PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/2D/particle_filter_2d.def
        )
    else()
        set_target_properties(particle_filter_2d_shared PROPERTIES PREFIX "")
    endif()
endif()

#───────────────────────────────────────────────────────────────────────────────
# Kelly Integration (disabled - not wired up yet)
#───────────────────────────────────────────────────────────────────────────────

# if(BUILD_KELLY)
#     add_library(kelly_integration INTERFACE)
#     
#     target_include_directories(kelly_integration INTERFACE
#         ${CMAKE_CURRENT_SOURCE_DIR}/kelly_integration
#     )
# 
#     # Kelly depends on the particle filters
#     if(BUILD_1D)
#         target_link_libraries(kelly_integration INTERFACE particle_filter)
#     endif()
#     if(BUILD_2D)
#         target_link_libraries(kelly_integration INTERFACE particle_filter_2d)
#     endif()
# endif()

#───────────────────────────────────────────────────────────────────────────────
# Benchmarks
#───────────────────────────────────────────────────────────────────────────────

if(BUILD_TESTS)
    enable_testing()

    # 1D benchmark
    if(BUILD_1D)
        add_executable(bench_pf1d
            test/example_usage.c
        )
        target_link_libraries(bench_pf1d PRIVATE
            particle_filter
        )
        if(UNIX)
            target_link_libraries(bench_pf1d PRIVATE m)
        endif()
        add_test(NAME bench_pf1d COMMAND bench_pf1d 4000 10000)
    endif()

    # 2D benchmark
    if(BUILD_2D)
        add_executable(bench_pf2d
            test/example_usage_2d.c
        )
        target_link_libraries(bench_pf2d PRIVATE
            particle_filter_2d
        )
        if(UNIX)
            target_link_libraries(bench_pf2d PRIVATE m)
        endif()
        add_test(NAME bench_pf2d COMMAND bench_pf2d 4000 10000)

        # HIGH_N benchmark (separate executable)
        if(PF2D_HIGH_N)
            add_executable(bench_pf2d_high_n
                test/example_usage_2d.c
            )
            target_link_libraries(bench_pf2d_high_n PRIVATE
                particle_filter_2d
            )
            if(UNIX)
                target_link_libraries(bench_pf2d_high_n PRIVATE m)
            endif()
            add_test(NAME bench_pf2d_high_n COMMAND bench_pf2d_high_n 50000 5000)
        endif()
    endif()
endif()

#───────────────────────────────────────────────────────────────────────────────
# Unit Tests
#───────────────────────────────────────────────────────────────────────────────

enable_testing()

# PF2D integration tests - requires MKL
if(BUILD_2D)
    add_executable(test_pf2d
        test/test_particle_filter_2d.c
    )

    target_include_directories(test_pf2d PRIVATE 
        ${CMAKE_SOURCE_DIR}/2D
    )

    target_link_libraries(test_pf2d PRIVATE 
        particle_filter_2d
    )

    if(UNIX)
        target_link_libraries(test_pf2d PRIVATE m)
    endif()

    add_test(NAME particle_filter_tests COMMAND test_pf2d)
endif()

# Adaptive self-calibration tests - standalone, no MKL dependency
add_executable(test_pf2d_adaptive
    test/test_pf2d_adaptive.c
)

target_compile_options(test_pf2d_adaptive PRIVATE
    $<$<C_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall -Wextra -O2>
)

if(UNIX)
    target_link_libraries(test_pf2d_adaptive PRIVATE m)
endif()

add_test(NAME adaptive_calibration_tests COMMAND test_pf2d_adaptive)

# PMMH parameter estimation tests - standalone, no MKL dependency
add_executable(test_pf2d_pmmh
    test/test_pf2d_pmmh.c
)

target_compile_options(test_pf2d_pmmh PRIVATE
    $<$<C_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall -Wextra -O2>
)

if(UNIX)
    target_link_libraries(test_pf2d_pmmh PRIVATE m)
endif()

add_test(NAME pmmh_estimation_tests COMMAND test_pf2d_pmmh)

# BOCPD + PMMH integration example - standalone
add_executable(example_bocpd_pmmh
    test/example_bocpd_pmmh.c
)

target_compile_options(example_bocpd_pmmh PRIVATE
    $<$<C_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall -Wextra -O2>
)

if(UNIX)
    target_link_libraries(example_bocpd_pmmh PRIVATE m)
endif()

add_test(NAME bocpd_pmmh_integration COMMAND example_bocpd_pmmh)

# Realistic scenario tests (Monte Carlo) - standalone
add_executable(test_realistic_scenarios
    test/test_realistic_scenarios.c
)

target_compile_options(test_realistic_scenarios PRIVATE
    $<$<C_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall -Wextra -O2>
)

if(UNIX)
    target_link_libraries(test_realistic_scenarios PRIVATE m)
endif()

add_test(NAME realistic_scenario_tests COMMAND test_realistic_scenarios)

#───────────────────────────────────────────────────────────────────────────────
# Install
#───────────────────────────────────────────────────────────────────────────────

# Libraries
if(BUILD_1D)
    install(TARGETS particle_filter particle_filter_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

if(BUILD_2D)
    install(TARGETS particle_filter_2d particle_filter_2d_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

# Headers
if(BUILD_1D)
    install(FILES 1D/particle_filter.h
        DESTINATION include/particle_filter
    )
endif()

if(BUILD_2D)
    install(FILES 2D/particle_filter_2d.h
        DESTINATION include/particle_filter
    )
endif()

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== Particle Filter Configuration ===")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:         ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "MKL threading:    ${MKL_THREADING}")
message(STATUS "Build 1D:         ${BUILD_1D}")
message(STATUS "Build 2D:         ${BUILD_2D}")
message(STATUS "Build tests:      ${BUILD_TESTS}")
message(STATUS "Float precision:  ${PF2D_USE_FLOAT}")
message(STATUS "HIGH_N mode:      ${PF2D_HIGH_N}")
if(MSVC)
    message(STATUS "Platform:         Windows (MSVC + Intel OpenMP)")
else()
    message(STATUS "Platform:         Unix-like")
endif()
message(STATUS "")