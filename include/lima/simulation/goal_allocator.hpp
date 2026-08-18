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

// Lifelong task stream. Without a fixed external sequence, goals are drawn
// from uniquely owned interior cells. Fixed sequences may share physical
// boundary workstations across live tasks because goal visits do not imply
// stay-at-goal occupancy; the executor still serializes actual cell use.
// Owns a dedicated RNG so task arrivals never perturb the simulator's shared
// random stream.
class GoalAllocator {
public:
    GoalAllocator(const GridMap& map, std::span<const Agent> agents, std::uint64_t seed,
                  std::span<const std::vector<Coord>> fixed_sequences = {});

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
    std::vector<std::vector<CellId>> fixed_sequences_;
    std::vector<std::size_t> fixed_cursors_;
    std::mt19937_64 rng_;
};

}  // namespace lima
