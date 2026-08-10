#include "lima/intersection/deadlock_detector.hpp"

#include <array>
#include <functional>
#include <unordered_map>

namespace lima {

std::vector<IntersectionIntent> collect_intents(const Intersection& intersection, const std::span<const Agent> agents) {
    std::vector<AgentId> members;
    members.reserve(agents.size());
    for (const Agent& agent : agents) if (agent.active) members.push_back(agent.id);
    return collect_intents(intersection, agents, members);
}

std::vector<IntersectionIntent> collect_intents(const Intersection& intersection, const std::span<const Agent> agents,
                                                const std::span<const AgentId> members) {
    std::vector<IntersectionIntent> result;
    result.reserve(members.size());
    for (const AgentId id : members) {
        if (id < 0 || static_cast<std::size_t>(id) >= agents.size()) continue;
        const Agent& agent = agents[static_cast<std::size_t>(id)];
        if (!agent.active) continue;
        const Direction current = intersection.direction_of(agent.position);
        if (current == Direction::None) continue;
        Direction exit = current;
        for (std::size_t i = agent.route_cursor; i + 1 < agent.route.size(); ++i) {
            if (agent.route[i] != intersection.center) continue;
            const Direction candidate = intersection.direction_of(agent.route[i + 1]);
            if (candidate != Direction::None && candidate != Direction::Center) exit = candidate;
            break;
        }
        result.push_back({agent.id, current, exit, agent.position, agent.intended_cell()});
    }
    return result;
}

bool has_intersection_deadlock(const Intersection& intersection, const std::span<const IntersectionIntent> intents) {
    std::array<std::array<bool, 4>, 4> graph{};
    std::unordered_map<CellId, const IntersectionIntent*> by_position;
    Direction center_exit = Direction::None;
    for (const auto& intent : intents) {
        by_position[intent.position] = &intent;
        if (intent.current == Direction::Center) center_exit = intent.exit;
        const auto from = static_cast<std::size_t>(intent.current);
        const auto to = static_cast<std::size_t>(intent.exit);
        if (from < 4 && to < 4 && from != to) graph[from][to] = true;
    }

    std::array<std::uint8_t, 4> color{};
    std::function<bool(std::size_t)> visit = [&](const std::size_t u) {
        color[u] = 1;
        for (std::size_t v = 0; v < 4; ++v) {
            if (!graph[u][v]) continue;
            if (color[v] == 1 || (color[v] == 0 && visit(v))) return true;
        }
        color[u] = 2;
        return false;
    };
    for (std::size_t d = 0; d < 4; ++d) if (color[d] == 0 && visit(d)) return true;

    if (static_cast<std::size_t>(center_exit) < 4) {
        for (const auto& intent : intents) {
            if (intent.current == center_exit && intent.exit != intent.current) return true;
        }
    }

    for (const auto& a : intents) {
        for (const auto& b : intents) {
            if (a.agent >= b.agent) continue;
            if (a.next == b.position && b.next == a.position && a.position != a.next) return true;
        }
    }

    for (std::size_t d = 0; d < 4; ++d) {
        const auto& arm = intersection.arms[d];
        for (std::size_t i = 0; i + 1 < arm.size(); ++i) {
            const auto front = by_position.find(arm[i]);
            const auto behind = by_position.find(arm[i + 1]);
            if (front == by_position.end() || behind == by_position.end()) continue;
            if (front->second->exit == static_cast<Direction>(d) && behind->second->exit != static_cast<Direction>(d)) return true;
        }
    }
    return false;
}

}  // namespace lima
