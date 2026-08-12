#pragma once

#include "lima/scheduling/stack_rearrangement.hpp"

#include <cstddef>
#include <vector>

namespace lima {

struct BeamSearchOptions {
    std::size_t beam_width{2'048};
    std::size_t max_depth{512};
    std::size_t max_expanded_nodes{1'000'000};
};

// Stack entries are target stack indices and are ordered bottom-to-top.
std::vector<StackMove> solve_stack_rearrangement_beam(
    const std::vector<std::vector<int>>& stacks,
    const std::vector<int>& capacities,
    BeamSearchOptions options = {});

}  // namespace lima
