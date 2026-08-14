#include "lima/simulation/pibt_resolver.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace lima {

PibtResolver::PibtResolver(const GridMap& map, const std::uint64_t seed, const bool age_rate,
                           const bool relaxed_pass)
    : map_(map), rng_(seed ^ 0x9e3779b97f4a7c15ULL), age_rate_(age_rate),
      relaxed_pass_(relaxed_pass),
      reserved_by_(static_cast<std::size_t>(map.cell_count()), kNoAgent) {}

void PibtResolver::resize(const std::size_t agent_count) {
    const std::size_t previous = priority_age_.size();
    priority_age_.resize(agent_count);
    initial_distance_.resize(agent_count);
    priority_rank_.resize(agent_count);
    next_positions_.resize(agent_count, kInvalidCell);
    order_.reserve(agent_count);

    if (previous == agent_count) return;
    std::vector<std::uint32_t> ranks(agent_count);
    std::iota(ranks.begin(), ranks.end(), 0U);
    std::shuffle(ranks.begin(), ranks.end(), rng_);
    for (std::size_t i = 0; i < agent_count; ++i) priority_rank_[i] = ranks[i];
}

int PibtResolver::candidate_distance(const Agent& agent, const CellId candidate) const noexcept {
    // The next cell of the immutable global route is the local PIBT target.
    // While an agent is outside that route, this makes it return to the same
    // route instead of replacing the remaining global plan.
    const Coord from = map_.coord(candidate);
    const Coord target = map_.coord(agent.intended_cell());
    return std::abs(from.x - target.x) + std::abs(from.y - target.y);
}

bool PibtResolver::assign(const AgentId id) {
    if (try_candidates(id, false)) return true;
    // Last-resort pass: an agent that found no strict candidate would be
    // frozen in place, so it may now consider cells its caller permits only
    // under relaxation.  Cells already attempted in the strict pass stay
    // reserved by this agent and are therefore not retried.
    if (relaxed_pass_ && try_candidates(id, true)) return true;

    // Failed inheritance: the blocker remains in place. As in the original
    // PIBT procedure this overwrites the parent's provisional reservation of
    // the same current cell, forcing the parent to backtrack to another node.
    const auto agent_index = static_cast<std::size_t>(id);
    next_positions_[agent_index] = agents_[agent_index].position;
    reserved_by_[static_cast<std::size_t>(agents_[agent_index].position)] = id;
    return false;
}

bool PibtResolver::try_candidates(const AgentId id, const bool relaxed) {
    const auto agent_index = static_cast<std::size_t>(id);
    const Agent& agent = agents_[agent_index];

    std::array<CellId, 5> candidates{};
    std::size_t candidate_count = 0;
    candidates[candidate_count++] = agent.position;
    for (const CellId neighbor : map_.neighbors(agent.position)) candidates[candidate_count++] = neighbor;

    std::shuffle(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count), rng_);
    std::stable_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count),
        [&](const CellId lhs, const CellId rhs) {
            return candidate_distance(agent, lhs) < candidate_distance(agent, rhs);
        });

    for (std::size_t i = 0; i < candidate_count; ++i) {
        const CellId candidate = candidates[i];
        if (!(*candidate_allowed_)(id, candidate, relaxed)) continue;
        if (reserved_by_[static_cast<std::size_t>(candidate)] != kNoAgent) continue;
        const AgentId occupant = occupancy_[static_cast<std::size_t>(candidate)];
        if (occupant != kNoAgent
            && next_positions_[static_cast<std::size_t>(occupant)] == agent.position) {
            continue;
        }

        // Match the original PIBT backtracking rule: attempted destinations
        // stay reserved for this timestep. This prevents repeated exploration
        // of the same failed dependency branch and keeps the local search
        // bounded by the small candidate set.
        next_positions_[agent_index] = candidate;
        reserved_by_[static_cast<std::size_t>(candidate)] = id;
        if (occupant != kNoAgent && occupant != id) {
            const auto occupant_index = static_cast<std::size_t>(occupant);
            if (eligible_[occupant_index] == 0) continue;
            if (next_positions_[occupant_index] == kInvalidCell && !assign(occupant)) continue;
        }
        return true;
    }
    return false;
}

void PibtResolver::resolve(const std::span<const Agent> agents,
                           const std::span<const AgentId> occupancy,
                           const std::span<const std::uint8_t> eligible,
                           const std::span<const std::uint8_t> priority_class,
                           const CandidateAllowed& candidate_allowed,
                           std::vector<CellId>& next_positions) {
    if (occupancy.size() != static_cast<std::size_t>(map_.cell_count()))
        throw std::invalid_argument("PIBT occupancy size mismatch");
    if (eligible.size() != agents.size())
        throw std::invalid_argument("PIBT eligibility size mismatch");
    if (priority_class.size() != agents.size())
        throw std::invalid_argument("PIBT priority-class size mismatch");

    resize(agents.size());
    agents_ = agents;
    occupancy_ = occupancy;
    eligible_ = eligible;
    priority_class_ = priority_class;
    candidate_allowed_ = &candidate_allowed;
    std::fill(next_positions_.begin(), next_positions_.end(), kInvalidCell);
    std::fill(reserved_by_.begin(), reserved_by_.end(), kNoAgent);
    order_.clear();

    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (initial_distance_[i] == 0 && agents[i].route.size() > 1)
            initial_distance_[i] = static_cast<std::uint32_t>(agents[i].route.size() - 1);
        if (agents[i].active && agents[i].position != agents[i].goal)
            priority_age_[i] += age_rate_ ? 1U + (priority_rank_[i] & 7U) : 1U;
        else priority_age_[i] = 0;
        if (eligible[i] != 0) order_.push_back(static_cast<AgentId>(i));
        else if (agents[i].active)
            reserved_by_[static_cast<std::size_t>(agents[i].position)] = static_cast<AgentId>(i);
    }
    std::sort(order_.begin(), order_.end(), [&](const AgentId lhs, const AgentId rhs) {
        const auto l = static_cast<std::size_t>(lhs);
        const auto r = static_cast<std::size_t>(rhs);
        if (priority_class_[l] != priority_class_[r]) return priority_class_[l] > priority_class_[r];
        if (priority_age_[l] != priority_age_[r]) return priority_age_[l] > priority_age_[r];
        if (initial_distance_[l] != initial_distance_[r]) return initial_distance_[l] > initial_distance_[r];
        return priority_rank_[l] < priority_rank_[r];
    });

    for (const AgentId id : order_) {
        const auto index = static_cast<std::size_t>(id);
        if (next_positions_[index] == kInvalidCell) static_cast<void>(assign(id));
    }

    next_positions.assign(agents.size(), kInvalidCell);
    for (const AgentId id : order_) {
        const auto index = static_cast<std::size_t>(id);
        next_positions[index] = next_positions_[index];
    }
}

}  // namespace lima
