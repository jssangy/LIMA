#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace lima {

using StackMove = std::pair<int, int>;

struct IdaStarOptions {
    std::size_t max_iterations{1'000'000};
    bool deterministic_move{true};
};

// Stack entries are target stack indices and are ordered bottom-to-top.
std::vector<StackMove> solve_stack_rearrangement(
    const std::vector<std::vector<int>>& stacks,
    const std::vector<int>& capacities,
    IdaStarOptions options = {});

}  // namespace lima

