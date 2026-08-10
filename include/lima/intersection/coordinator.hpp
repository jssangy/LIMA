#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/deadlock_detector.hpp"

#include <cstdint>
#include <array>
#include <span>

namespace lima {

class IntersectionCoordinator {
public:
    bool schedule(const Intersection& intersection, std::span<Agent> agents,
                  std::span<const IntersectionIntent> intents, const std::array<int, 4>& stack_quotas,
                  std::int32_t group_id) const;
};

}  // namespace lima
