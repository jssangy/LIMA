#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/topology.hpp"

#include <span>
#include <vector>

namespace lima {

struct IntersectionIntent {
    AgentId agent{kNoAgent};
    Direction current{Direction::None};
    Direction exit{Direction::None};
    CellId position{kInvalidCell};
    CellId next{kInvalidCell};
};

std::vector<IntersectionIntent> collect_intents(const Intersection& intersection, std::span<const Agent> agents);
std::vector<IntersectionIntent> collect_intents(const Intersection& intersection, std::span<const Agent> agents,
                                                std::span<const AgentId> members);
void collect_intents(const Intersection& intersection, std::span<const Agent> agents,
                     std::span<const AgentId> members, std::vector<IntersectionIntent>& result);
bool has_intersection_deadlock(const Intersection& intersection, std::span<const IntersectionIntent> intents);

}  // namespace lima
