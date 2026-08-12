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

class GoalAllocator {
public:
    GoalAllocator(const GridMap& map, std::span<const Agent> agents, std::uint64_t seed);

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
