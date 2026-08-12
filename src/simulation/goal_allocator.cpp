#include "lima/simulation/goal_allocator.hpp"

#include "lima/core/grid_map.hpp"
#include "lima/io/scenario_loader.hpp"

#include <algorithm>
#include <stdexcept>

namespace lima {

GoalAllocator::GoalAllocator(const GridMap& map, const std::span<const Agent> agents,
                             const std::uint64_t seed)
    : candidates_(make_goal_candidates(map)),
      owner_(static_cast<std::size_t>(map.cell_count()), kNoAgent), rng_(seed) {
    free_and_empty_.reserve(candidates_.size());
    free_but_occupied_.reserve(candidates_.size());
    for (const Agent& agent : agents) {
        AgentId& owner = owner_[static_cast<std::size_t>(agent.goal)];
        if (owner != kNoAgent) throw std::runtime_error("lifelong mode requires unique initial goals");
        owner = agent.id;
    }
}

std::optional<CellId> GoalAllocator::reassign(
    const AgentId agent, const CellId previous_goal, const CellId current,
    const std::span<const AgentId> occupancy,
    const std::function<bool(CellId)>& acceptable) {
    AgentId& previous_owner = owner_[static_cast<std::size_t>(previous_goal)];
    if (previous_owner != agent)
        throw std::logic_error("lifelong goal ownership is inconsistent");
    previous_owner = kNoAgent;

    free_and_empty_.clear();
    free_but_occupied_.clear();
    for (const CellId candidate : candidates_) {
        if (candidate == current || owner_[static_cast<std::size_t>(candidate)] != kNoAgent) continue;
        if (occupancy[static_cast<std::size_t>(candidate)] == kNoAgent)
            free_and_empty_.push_back(candidate);
        else free_but_occupied_.push_back(candidate);
    }

    std::shuffle(free_and_empty_.begin(), free_and_empty_.end(), rng_);
    std::shuffle(free_but_occupied_.begin(), free_but_occupied_.end(), rng_);
    for (const auto* pool : {&free_and_empty_, &free_but_occupied_}) {
        for (const CellId goal : *pool) {
            if (!acceptable(goal)) continue;
            owner_[static_cast<std::size_t>(goal)] = agent;
            return goal;
        }
    }
    previous_owner = agent;
    return std::nullopt;
}

}  // namespace lima
