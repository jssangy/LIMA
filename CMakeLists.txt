cmake_minimum_required(VERSION 3.20)
project(lima LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LIMA_ENABLE_LTO "Enable link-time optimization for release builds" ON)
option(LIMA_BUILD_GUI "Build the SDL2 graphical viewer when SDL2 is available" ON)

add_library(lima_core
    src/core/grid_map.cpp
    src/io/scenario_loader.cpp
    src/io/solution_trace.cpp
    src/planning/planner.cpp
    src/intersection/topology.cpp
    src/intersection/deadlock_detector.cpp
    src/intersection/coordinator.cpp
    src/scheduling/ida_star.cpp
    src/simulation/simulator.cpp
)
target_include_directories(lima_core PUBLIC include)
target_compile_options(lima_core PRIVATE -Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(lima_core PRIVATE -O3 -DNDEBUG)
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
