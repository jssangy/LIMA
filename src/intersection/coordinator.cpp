#include "lima/intersection/coordinator.hpp"

#include "lima/scheduling/ida_star.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace lima {
namespace {

using Lane = std::vector<std::optional<AgentId>>;

Agent* find_agent(const AgentId id, const std::span<Agent> agents) {
    for (Agent& agent : agents) if (agent.id == id) return &agent;
    return nullptr;
}

bool adjacent_or_equal(const CellId a, const CellId b, const Intersection& intersection) {
    if (a == b) return true;
    if (a == intersection.center || b == intersection.center) {
        const CellId other = a == intersection.center ? b : a;
        for (const auto& arm : intersection.arms) if (!arm.empty() && arm.front() == other) return true;
    }
    for (const auto& arm : intersection.arms) {
        for (std::size_t i = 0; i + 1 < arm.size(); ++i) {
            if ((arm[i] == a && arm[i + 1] == b) || (arm[i] == b && arm[i + 1] == a)) return true;
        }
    }
    return false;
}

}  // namespace

bool IntersectionCoordinator::schedule(const Intersection& intersection, const std::span<Agent> agents,
                                       const std::span<const IntersectionIntent> intents,
                                       const std::array<int, 4>& stack_quotas,
                                       const std::int32_t group_id) const {
    std::vector<Direction> directions;
    for (std::size_t d = 0; d < 4; ++d) if (!intersection.arms[d].empty()) directions.push_back(static_cast<Direction>(d));
    if (directions.size() < 3 || intents.size() < 2) return false;

    std::unordered_map<AgentId, Direction> desired_direction;
    std::array<Lane, 4> current;
    for (const Direction d : directions) current[static_cast<std::size_t>(d)].resize(intersection.arms[static_cast<std::size_t>(d)].size());
    std::optional<AgentId> center_agent;
    for (const auto& intent : intents) {
        if (find_agent(intent.agent, agents) == nullptr) return false;
        desired_direction[intent.agent] = intent.exit;
        if (intent.current == Direction::Center) {
            if (center_agent) return false;
            center_agent = intent.agent;
            continue;
        }
        const auto d = static_cast<std::size_t>(intent.current);
        if (d >= 4) return false;
        const auto& cells = intersection.arms[d];
        const auto found = std::find(cells.begin(), cells.end(), intent.position);
        if (found == cells.end()) return false;
        current[d][static_cast<std::size_t>(found - cells.begin())] = intent.agent;
    }

    std::array<Lane, 4> target;
    for (const Direction direction : directions) {
        const auto d = static_cast<std::size_t>(direction);
        target[d].resize(current[d].size());
        std::size_t write = 0;
        for (const auto id : current[d]) if (id) target[d][write++] = id;
    }
    if (center_agent) {
        Direction host = desired_direction[*center_agent];
        auto has_space = [&](const Direction d) {
            if (static_cast<std::size_t>(d) >= 4 || target[static_cast<std::size_t>(d)].empty()) return false;
            return !target[static_cast<std::size_t>(d)].back().has_value();
        };
        if (!has_space(host)) {
            std::size_t best_count = std::numeric_limits<std::size_t>::max();
            host = Direction::None;
            for (const Direction d : directions) {
                const auto count = static_cast<std::size_t>(std::count_if(target[static_cast<std::size_t>(d)].begin(),
                    target[static_cast<std::size_t>(d)].end(), [](const auto id) { return id.has_value(); }));
                if (count < target[static_cast<std::size_t>(d)].size() && count < best_count) {
                    best_count = count;
                    host = d;
                }
            }
        }
        if (!has_space(host)) return false;
        auto& lane = target[static_cast<std::size_t>(host)];
        for (std::size_t i = lane.size() - 1; i > 0; --i) lane[i] = lane[i - 1];
        lane[0] = center_agent;
    }

    std::unordered_map<AgentId, std::vector<CellId>> paths;
    std::unordered_map<AgentId, std::pair<Direction, std::size_t>> current_location;
    std::unordered_map<AgentId, std::pair<Direction, std::size_t>> target_location;
    for (const auto& intent : intents) paths[intent.agent] = {intent.position};
    for (const Direction direction : directions) {
        const auto d = static_cast<std::size_t>(direction);
        for (std::size_t i = 0; i < current[d].size(); ++i) if (current[d][i]) current_location[*current[d][i]] = {direction, i};
        for (std::size_t i = 0; i < target[d].size(); ++i) if (target[d][i]) target_location[*target[d][i]] = {direction, i};
    }

    std::size_t prestage_steps = center_agent ? 1 : 0;
    for (const auto& [id, source] : current_location) {
        const auto destination = target_location.at(id);
        if (source.first != destination.first) return false;
        const auto distance = source.second > destination.second ? source.second - destination.second : destination.second - source.second;
        prestage_steps = std::max(prestage_steps, distance);
    }
    for (std::size_t step = 1; step <= prestage_steps; ++step) {
        for (const auto& intent : intents) {
            CellId position = paths[intent.agent].back();
            if (center_agent && intent.agent == *center_agent) {
                const auto [direction, index] = target_location.at(intent.agent);
                (void)index;
                if (step == 1) position = intersection.arms[static_cast<std::size_t>(direction)][0];
            } else {
                const auto [direction, from] = current_location.at(intent.agent);
                const auto to = target_location.at(intent.agent).second;
                std::size_t index = from;
                if (from < to) index = std::min(to, from + step);
                else if (from > to) index = from - std::min(from - to, step);
                position = intersection.arms[static_cast<std::size_t>(direction)][index];
            }
            paths[intent.agent].push_back(position);
        }
    }

    std::unordered_map<Direction, int> direction_index;
    for (std::size_t i = 0; i < directions.size(); ++i) direction_index[directions[i]] = static_cast<int>(i);
    std::vector<std::vector<AgentId>> stacks(directions.size());
    std::vector<std::vector<int>> solver_stacks(directions.size());
    std::vector<int> capacities;
    for (std::size_t i = 0; i < directions.size(); ++i) {
        const auto d = static_cast<std::size_t>(directions[i]);
        capacities.push_back(static_cast<int>(target[d].size()));
        for (auto it = target[d].rbegin(); it != target[d].rend(); ++it) if (*it) {
            stacks[i].push_back(**it);
            Direction desired = desired_direction[**it];
            if (!direction_index.contains(desired)) desired = directions[i];
            solver_stacks[i].push_back(direction_index[desired]);
        }
    }

    std::vector<StackMove> actions;
    try {
        actions = solve_stack_rearrangement(solver_stacks, capacities);
    } catch (const std::exception&) {
        return false;
    }

    // Apply the neighbor-capacity quotas as the same lightweight post-process
    // used by the Python scheduler. Overflow robots may be parked on another arm.
    auto final_types = solver_stacks;
    for (const auto& [src, dst] : actions) {
        if (final_types[src].empty() || final_types[dst].size() >= static_cast<std::size_t>(capacities[dst])) return false;
        const int item = final_types[src].back();
        final_types[src].pop_back();
        final_types[dst].push_back(item);
    }
    std::vector<int> quotas;
    quotas.reserve(directions.size());
    for (std::size_t i = 0; i < directions.size(); ++i) {
        const int quota = stack_quotas[static_cast<std::size_t>(directions[i])];
        quotas.push_back(std::clamp(quota, 0, capacities[i]));
    }
    for (std::size_t guard = 0; guard < 4096; ++guard) {
        int src = -1;
        int largest_overflow = 0;
        for (std::size_t i = 0; i < final_types.size(); ++i) {
            const int overflow = static_cast<int>(final_types[i].size()) - quotas[i];
            if (overflow > largest_overflow) {
                largest_overflow = overflow;
                src = static_cast<int>(i);
            }
        }
        if (src < 0) break;
        int dst = -1;
        int best_slack = 0;
        for (std::size_t i = 0; i < final_types.size(); ++i) {
            if (static_cast<int>(i) == src || final_types[i].size() >= static_cast<std::size_t>(capacities[i])) continue;
            const int slack = quotas[i] - static_cast<int>(final_types[i].size());
            if (slack > best_slack) {
                best_slack = slack;
                dst = static_cast<int>(i);
            }
        }
        if (dst < 0 || final_types[src].empty()) return false;
        const int item = final_types[src].back();
        final_types[src].pop_back();
        final_types[dst].push_back(item);
        actions.emplace_back(src, dst);
    }

    const auto append_snapshot = [&](const std::optional<AgentId> center) {
        std::unordered_map<AgentId, CellId> positions;
        if (center) positions[*center] = intersection.center;
        for (std::size_t i = 0; i < directions.size(); ++i) {
            const auto& cells = intersection.arms[static_cast<std::size_t>(directions[i])];
            for (std::size_t near = 0; near < stacks[i].size(); ++near) positions[stacks[i][stacks[i].size() - 1 - near]] = cells[near];
        }
        for (const auto& intent : intents) {
            const auto found = positions.find(intent.agent);
            if (found == positions.end()) throw std::logic_error("scheduled agent vanished from stack state");
            paths[intent.agent].push_back(found->second);
        }
    };

    std::optional<AgentId> center;
    int pending_destination = -1;
    for (const auto& [src, dst] : actions) {
        if (src < 0 || dst < 0 || static_cast<std::size_t>(src) >= stacks.size()
            || static_cast<std::size_t>(dst) >= stacks.size()) return false;

        // Match the Python scheduler: finish the previous center push and pull the
        // next source in the same logical frame.
        if (center) {
            if (pending_destination < 0
                || stacks[static_cast<std::size_t>(pending_destination)].size()
                    >= static_cast<std::size_t>(capacities[static_cast<std::size_t>(pending_destination)])) return false;
            stacks[static_cast<std::size_t>(pending_destination)].push_back(*center);
            center.reset();
        }
        if (stacks[src].empty()) return false;
        const AgentId moving = stacks[src].back();
        stacks[src].pop_back();
        center = moving;
        pending_destination = dst;
        append_snapshot(center);
    }
    if (center) {
        if (stacks[static_cast<std::size_t>(pending_destination)].size()
            >= static_cast<std::size_t>(capacities[static_cast<std::size_t>(pending_destination)])) return false;
        stacks[static_cast<std::size_t>(pending_destination)].push_back(*center);
        append_snapshot(std::nullopt);
    }

    if (actions.empty() && prestage_steps == 0) return false;
    std::vector<std::pair<Agent*, std::vector<CellId>>> assignments;
    assignments.reserve(intents.size());
    for (const auto& intent : intents) {
        Agent* agent = find_agent(intent.agent, agents);
        if (agent == nullptr) return false;
        auto& path = paths[intent.agent];

        // Keep every participant on the complete synchronized schedule.  The
        // former post-process both truncated intended-exit paths and removed
        // straight-through holders that never touched the center; either form
        // of early release could drop group isolation before the snapshot ends.
        if (path.size() < 2) return false;
        for (std::size_t i = 0; i + 1 < path.size(); ++i) if (!adjacent_or_equal(path[i], path[i + 1], intersection)) return false;
        assignments.emplace_back(agent, std::move(path));
    }
    if (assignments.empty()) return false;
    // Commit only after every shortened path has been validated.  A late
    // validation failure must never leave a subset of the group scheduled.
    for (auto& [agent, path] : assignments) {
        agent->schedule_route = std::move(path);
        agent->schedule_cursor = 0;
        agent->schedule_group = group_id;
    }
    return true;
}

}  // namespace lima
