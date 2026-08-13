cmake_minimum_required(VERSION 3.20)
project(lima LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LIMA_ENABLE_LTO "Enable link-time optimization for release builds" ON)
option(LIMA_NATIVE_OPTIMIZATION "Optimize release builds for the current CPU" OFF)
option(LIMA_BUILD_GUI "Build the SDL2 graphical viewer when SDL2 is available" ON)

# Build provenance: regenerated on every build so experiment summaries always
# carry the exact commit hash (plus -dirty when the tree has local edits).
set(LIMA_VERSION_HEADER ${CMAKE_BINARY_DIR}/generated/lima_version.hpp)
add_custom_target(lima_version
    COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -DOUT=${LIMA_VERSION_HEADER}
            -P ${CMAKE_SOURCE_DIR}/cmake/version.cmake
    BYPRODUCTS ${LIMA_VERSION_HEADER}
    COMMENT "Stamping git commit into lima_version.hpp")

add_library(lima_core
    src/core/grid_map.cpp
    src/io/scenario_loader.cpp
    src/io/solution_trace.cpp
    src/planning/planner.cpp
    src/intersection/topology.cpp
    src/intersection/deadlock_detector.cpp
    src/intersection/coordinator.cpp
    src/intersection/gating.cpp
    src/scheduling/beam_search.cpp
    src/scheduling/ida_star.cpp
    src/scheduling/solver.cpp
    src/simulation/goal_allocator.cpp
    src/simulation/metrics.cpp
    src/simulation/simulator.cpp
    src/simulation/trace.cpp
)
target_include_directories(lima_core PUBLIC include)
target_compile_options(lima_core PRIVATE -Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(lima_core PRIVATE -O3 -DNDEBUG)
    if(LIMA_NATIVE_OPTIMIZATION)
        target_compile_options(lima_core PRIVATE -march=native)
    endif()
    if(LIMA_ENABLE_LTO)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
        if(ipo_supported)
            set_property(TARGET lima_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
    endif()
endif()

add_executable(lima app/main.cpp)
target_link_libraries(lima PRIVATE lima_core)
add_dependencies(lima lima_version)
target_include_directories(lima PRIVATE ${CMAKE_BINARY_DIR}/generated)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(LIMA_NATIVE_OPTIMIZATION)
        target_compile_options(lima PRIVATE -march=native)
    endif()
    if(LIMA_ENABLE_LTO AND ipo_supported)
        set_property(TARGET lima PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

if(LIMA_BUILD_GUI)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(SDL2 QUIET IMPORTED_TARGET sdl2)
    endif()
    if(TARGET PkgConfig::SDL2)
        target_sources(lima PRIVATE src/viewer/viewer.cpp)
        target_link_libraries(lima PRIVATE PkgConfig::SDL2)
        target_compile_definitions(lima PRIVATE LIMA_HAS_SDL2=1)
    else()
        message(WARNING "SDL2 was not found; the lima executable will be headless only")
    endif()
endif()
