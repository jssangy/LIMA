#pragma once

#include "lima/simulation/simulator.hpp"
#include "lima/io/solution_trace.hpp"

#include <cstdint>

namespace lima {

struct ViewerOptions {
    double steps_per_second{20.0};
    std::uint64_t max_steps{100'000};
};

#ifdef LIMA_HAS_SDL2
int run_viewer(Simulator& simulator, ViewerOptions options, SolutionTrace* recorder = nullptr);
int run_replay(const GridMap& map, const SolutionTrace& trace, ViewerOptions options);
#endif

}  // namespace lima
