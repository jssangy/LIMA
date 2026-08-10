#include "lima/simulation/dependency_resolver.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace lima {
namespace {

bool higher_priority(const MoveIntent& lhs, const MoveIntent& rhs) {
    const bool lhs_scheduled = lhs.schedule_group != kNoGroup;
    const bool rhs_scheduled = rhs.schedule_group != kNoGroup;
    if (lhs_scheduled != rhs_scheduled) return lhs_scheduled;
    if (lhs.wait_steps != rhs.wait_steps) return lhs.wait_steps > rhs.wait_steps;
    return lhs.agent < rhs.agent;
}

}  // namespace

MoveResolution DependencyResolver::resolve(const std::span<const MoveIntent> intents) const {
    std::unordered_set<std::int32_t> banned_groups;
    std::vector<bool> final_result(intents.size(), false);
    std::vector<WaitReason> final_reasons(intents.size(), WaitReason::Dependency);

    for (;;) {
        std::vector<bool> candidate(intents.size(), false);
        std::vector<WaitReason> reasons(intents.size(), WaitReason::Dependency);
        std::vector<AgentId> occupancy(static_cast<std::size_t>(map_.cell_count()), kNoAgent);
        std::unordered_map<AgentId, std::size_t> by_agent;
        std::unordered_map<CellId, std::size_t> target_winner;
        for (std::size_t i = 0; i < intents.size(); ++i) {
            const auto& intent = intents[i];
            if (!map_.traversable(intent.from) || occupancy[static_cast<std::size_t>(intent.from)] != kNoAgent) {
                throw std::logic_error("invalid or duplicate current occupancy");
            }
            occupancy[static_cast<std::size_t>(intent.from)] = intent.agent;
            if (!by_agent.emplace(intent.agent, i).second) throw std::logic_error("duplicate agent intent");
        }

        for (std::size_t i = 0; i < intents.size(); ++i) {
            const auto& intent = intents[i];
            if (intent.schedule_group != kNoGroup && banned_groups.contains(intent.schedule_group)) continue;
            if (intent.from == intent.to) {
                candidate[i] = true;
                reasons[i] = WaitReason::None;
                continue;
            }
            if (!map_.traversable(intent.to)) continue;
            const auto& adjacent = map_.neighbors(intent.from);
            if (std::find(adjacent.begin(), adjacent.end(), intent.to) == adjacent.end()) continue;
            const auto [it, inserted] = target_winner.emplace(intent.to, i);
            if (inserted || higher_priority(intent, intents[it->second])) {
                if (!inserted) reasons[it->second] = WaitReason::VertexConflict;
                it->second = i;
            } else {
                reasons[i] = WaitReason::VertexConflict;
            }
        }
        for (const auto [cell, index] : target_winner) {
            (void)cell;
            candidate[index] = true;
        }

        // A direct edge swap is never committed.
        for (std::size_t i = 0; i < intents.size(); ++i) {
            if (!candidate[i]) continue;
            const AgentId occupant = occupancy[static_cast<std::size_t>(intents[i].to)];
            if (occupant == kNoAgent) continue;
            const std::size_t j = by_agent.at(occupant);
            if (j != i && candidate[j] && intents[j].to == intents[i].from) {
                candidate[i] = false;
                candidate[j] = false;
                reasons[i] = WaitReason::EdgeSwap;
                reasons[j] = WaitReason::EdgeSwap;
            }
        }

        enum class State : std::uint8_t { Unknown, Visiting, Approved, Rejected };
        std::vector<State> state(intents.size(), State::Unknown);
        std::function<bool(std::size_t)> approve = [&](const std::size_t i) -> bool {
            if (!candidate[i]) return false;
            if (intents[i].from == intents[i].to) return true;
            if (state[i] == State::Approved) return true;
            if (state[i] == State::Rejected) return false;
            if (state[i] == State::Visiting) {
                state[i] = State::Rejected;  // cycles require explicit intersection scheduling
                return false;
            }
            state[i] = State::Visiting;
            const AgentId occupant = occupancy[static_cast<std::size_t>(intents[i].to)];
            bool allowed = occupant == kNoAgent;
            if (occupant != kNoAgent) {
                const std::size_t dependency = by_agent.at(occupant);
                allowed = dependency != i && intents[dependency].from != intents[dependency].to && approve(dependency);
            }
            state[i] = allowed ? State::Approved : State::Rejected;
            return allowed;
        };

        std::vector<bool> result(intents.size(), false);
        for (std::size_t i = 0; i < intents.size(); ++i) {
            result[i] = approve(i);
            if (result[i]) reasons[i] = WaitReason::None;
            else if (candidate[i] && reasons[i] == WaitReason::None) reasons[i] = WaitReason::Dependency;
        }

        std::unordered_set<std::int32_t> newly_banned;
        for (std::size_t i = 0; i < intents.size(); ++i) {
            const auto group = intents[i].schedule_group;
            if (group != kNoGroup && intents[i].from != intents[i].to && !result[i]) newly_banned.insert(group);
        }
        const std::size_t before = banned_groups.size();
        banned_groups.insert(newly_banned.begin(), newly_banned.end());
        for (std::size_t i = 0; i < intents.size(); ++i) {
            if (intents[i].schedule_group != kNoGroup && banned_groups.contains(intents[i].schedule_group)) {
                reasons[i] = WaitReason::ScheduleGroup;
            }
        }
        final_result = std::move(result);
        final_reasons = std::move(reasons);
        if (banned_groups.size() == before) break;
    }

    validate_transition(intents, final_result);
    return {std::move(final_result), std::move(final_reasons)};
}

void DependencyResolver::validate_transition(const std::span<const MoveIntent> intents,
                                             const std::vector<bool>& approved) const {
    if (intents.size() != approved.size()) throw std::logic_error("transition size mismatch");
    std::vector<AgentId> final_occupancy(static_cast<std::size_t>(map_.cell_count()), kNoAgent);
    for (std::size_t i = 0; i < intents.size(); ++i) {
        const CellId final_cell = approved[i] ? intents[i].to : intents[i].from;
        if (!map_.traversable(final_cell)) throw std::logic_error("transition ends outside traversable map");
        auto& slot = final_occupancy[static_cast<std::size_t>(final_cell)];
        if (slot != kNoAgent) throw std::logic_error("vertex conflict survived arbitration");
        slot = intents[i].agent;
    }
    for (std::size_t i = 0; i < intents.size(); ++i) for (std::size_t j = i + 1; j < intents.size(); ++j) {
        if (approved[i] && approved[j] && intents[i].from == intents[j].to && intents[j].from == intents[i].to) {
            throw std::logic_error("edge swap survived arbitration");
        }
    }
}

}  // namespace lima
