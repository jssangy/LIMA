#pragma once

#include "lima/core/agent.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace lima {

class GridMap;

// Lifelong task stream: hands each completed agent a fresh goal drawn from
// the interior traversable non-S cells (S sinks are one-shot disappear
// exits, not persistent lifelong goals; the outermost row/col is excluded to
// match the scenario-generator start rule).  Goals handed out by the
// allocator are uniquely owned, so two live tasks never target the same
// cell; free cells that are currently unoccupied are preferred over
// free-but-occupied ones.  Owns a dedicated RNG so task arrivals never
// perturb the simulator's shared random stream.
class GoalAllocator {
public:
    GoalAllocator(const GridMap& map, std::span<const Agent> agents, std::uint64_t seed);

    // Returns the next goal for `agent`, or nullopt when no acceptable free
    // cell exists this step (the caller retries on a later step).  Ownership
    // of `previous_goal` is released; `current` is excluded from the draw.
    [[nodiscard]] std::optional<CellId> reassign(
        AgentId agent, CellId previous_goal, CellId current,
        std::span<const AgentId> occupancy,
        const std::function<bool(CellId)>& acceptable);

private:
    std::vector<CellId> candidates_;
    std::vector<AgentId> owner_;
    std::vector<CellId> free_and_empty_;
    std::vector<CellId> free_but_occupied_;
    std::mt19937_64 rng_;
};

}  // namespace lima
