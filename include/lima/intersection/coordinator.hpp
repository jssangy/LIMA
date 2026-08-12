#pragma once

#include "lima/core/agent.hpp"
#include "lima/intersection/deadlock_detector.hpp"
#include "lima/scheduling/solver.hpp"

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

// Telemetry for one scheduling attempt, consumed by the metrics layer.
struct ScheduleTelemetry {
    std::size_t intents{};
    SolverStats solver;
};

class IntersectionCoordinator {
public:
    explicit IntersectionCoordinator(StackSolver& solver) : solver_(solver) {}

    std::optional<std::vector<ScheduledPath>> schedule(
        const Intersection& intersection, std::span<const IntersectionIntent> intents,
        const std::array<int, 4>& stack_quotas, ScheduleTelemetry* telemetry = nullptr) const;

private:
    StackSolver& solver_;
};

}  // namespace lima
