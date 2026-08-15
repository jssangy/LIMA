#include "lima/simulation/goal_allocator.hpp"

#include "lima/core/grid_map.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace lima {

namespace {

// Lifelong goal pool (approved design, 2026-08-13 re-freeze): interior
// traversable non-S cells.  S cells are shared disappear-exits for the
// one-shot mode and must not serve as persistent lifelong goals; the
// outermost row/col is excluded to match the scenario-generator start rule.
// Mirrors make_goal_candidates from the coworker's scenario_loader, with the
// pool fixed to interior non-sink cells.
std::vector<CellId> make_goal_candidates(const GridMap& map) {
    const std::unordered_set<CellId> sinks(map.sink_cells().begin(), map.sink_cells().end());
    std::vector<CellId> candidates;
    candidates.reserve(map.traversable_cells().size());
    for (const CellId id : map.traversable_cells()) {
        const Coord c = map.coord(id);
        if (c.x == 0 || c.y == 0 || c.x + 1 == map.width() || c.y + 1 == map.height()) continue;
        if (sinks.contains(id)) continue;
        candidates.push_back(id);
    }
    if (candidates.empty())
        throw std::runtime_error("lifelong mode found no interior non-sink goal cells");
    return candidates;
}

}  // namespace

GoalAllocator::GoalAllocator(const GridMap& map, const std::span<const Agent> agents,
                             const std::uint64_t seed,
                             const std::span<const std::vector<Coord>> fixed_sequences)
    : candidates_(make_goal_candidates(map)),
      owner_(static_cast<std::size_t>(map.cell_count()), kNoAgent), rng_(seed) {
    free_and_empty_.reserve(candidates_.size());
    free_but_occupied_.reserve(candidates_.size());
    // Random scenarios draw initial goals with replacement, so several agents
    // may start out targeting the same cell.  The first agent claims the
    // ownership slot; the others simply do not own their initial goal and
    // acquire a uniquely owned one on their first reassignment.
    for (const Agent& agent : agents) {
        if (!agent.active) continue;
        AgentId& owner = owner_[static_cast<std::size_t>(agent.goal)];
        if (owner == kNoAgent) owner = agent.id;
    }
    if (!fixed_sequences.empty()) {
        if (fixed_sequences.size() != agents.size())
            throw std::runtime_error("lifelong goal sequence count does not match agent count");
        fixed_sequences_.resize(agents.size());
        fixed_cursors_.resize(agents.size(), 1);
        for (std::size_t i = 0; i < fixed_sequences.size(); ++i) {
            const auto& source = fixed_sequences[i];
            if (source.size() < 2)
                throw std::runtime_error("each lifelong goal sequence needs at least two goals");
            auto& destination = fixed_sequences_[i];
            destination.reserve(source.size());
            for (const Coord coordinate : source) {
                if (!map.in_bounds(coordinate) || !map.traversable(coordinate))
                    throw std::runtime_error("lifelong goal sequence contains a non-traversable cell");
                destination.push_back(map.cell(coordinate));
            }
            if (destination.front() != agents[i].goal)
                throw std::runtime_error("lifelong sequence first goal does not match initial task goal");
            for (std::size_t j = 0; j < destination.size(); ++j) {
                if (destination[j] == destination[(j + 1) % destination.size()])
                    throw std::runtime_error("lifelong goal sequence contains consecutive duplicate goals");
            }
        }
    }
}

std::optional<CellId> GoalAllocator::reassign(
    const AgentId agent, const CellId previous_goal, const CellId current,
    const std::span<const AgentId> occupancy,
    const std::function<bool(CellId)>& acceptable) {
    AgentId& previous_owner = owner_[static_cast<std::size_t>(previous_goal)];
    const bool owned = previous_owner == agent;
    if (owned) previous_owner = kNoAgent;

    if (!fixed_sequences_.empty()) {
        const auto index = static_cast<std::size_t>(agent);
        auto& cursor = fixed_cursors_[index];
        const CellId goal = fixed_sequences_[index][cursor];
        AgentId& next_owner = owner_[static_cast<std::size_t>(goal)];
        if (goal == current || next_owner != kNoAgent || !acceptable(goal)) {
            if (owned) previous_owner = agent;
            return std::nullopt;
        }
        next_owner = agent;
        cursor = (cursor + 1) % fixed_sequences_[index].size();
        return goal;
    }

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
    if (owned) previous_owner = agent;
    return std::nullopt;
}

}  // namespace lima
