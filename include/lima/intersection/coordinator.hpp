#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/deadlock_detector.hpp"

#include <cstdint>
#include <array>
#include <optional>
#include <span>
#include <vector>

namespace lima {

struct ScheduledPath {
    AgentId agent{kNoAgent};
    std::vector<CellId> path;
    CellId target_exit{kInvalidCell};
};

class IntersectionCoordinator {
public:
    std::optional<std::vector<ScheduledPath>> schedule(
        const Intersection& intersection, std::span<const IntersectionIntent> intents,
        const std::array<int, 4>& stack_quotas) const;
};

}  // namespace lima
